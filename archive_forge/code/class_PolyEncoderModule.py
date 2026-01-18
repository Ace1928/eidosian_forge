from typing import Any, Dict, Optional, Tuple
import torch
from parlai.core.opt import Opt
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .biencoder import AddLabelFixedCandsTRA
from .modules import (
from .transformer import TransformerRankerAgent
class PolyEncoderModule(torch.nn.Module):
    """
    Poly-encoder model.

    See https://arxiv.org/abs/1905.01969 for more details
    """

    def __init__(self, opt, dict_, null_idx):
        super(PolyEncoderModule, self).__init__()
        self.null_idx = null_idx
        self.encoder_ctxt = self.get_encoder(opt=opt, dict_=dict_, null_idx=null_idx, reduction_type=None, for_context=True)
        self.encoder_cand = self.get_encoder(opt=opt, dict_=dict_, null_idx=null_idx, reduction_type=opt['reduction_type'], for_context=False)
        self.type = opt['polyencoder_type']
        self.n_codes = opt['poly_n_codes']
        self.attention_type = opt['poly_attention_type']
        self.attention_num_heads = opt['poly_attention_num_heads']
        self.codes_attention_type = opt['codes_attention_type']
        self.codes_attention_num_heads = opt['codes_attention_num_heads']
        embed_dim = opt['embedding_size']
        if self.type == 'codes':
            codes = torch.empty(self.n_codes, embed_dim)
            codes = torch.nn.init.uniform_(codes)
            self.codes = torch.nn.Parameter(codes)
            if self.codes_attention_type == 'multihead':
                self.code_attention = MultiHeadAttention(self.codes_attention_num_heads, embed_dim, opt['dropout'])
            elif self.codes_attention_type == 'sqrt':
                self.code_attention = PolyBasicAttention(self.type, self.n_codes, dim=2, attn='sqrt', get_weights=False)
            elif self.codes_attention_type == 'basic':
                self.code_attention = PolyBasicAttention(self.type, self.n_codes, dim=2, attn='basic', get_weights=False)
        if self.attention_type == 'multihead':
            self.attention = MultiHeadAttention(self.attention_num_heads, opt['embedding_size'], opt['dropout'])
        else:
            self.attention = PolyBasicAttention(self.type, self.n_codes, dim=2, attn=self.attention_type, get_weights=False)

    def get_encoder(self, opt, dict_, null_idx, reduction_type, for_context: bool):
        """
        Return encoder, given options.

        :param opt:
            opt dict
        :param dict:
            dictionary agent
        :param null_idx:
            null/pad index into dict
        :param reduction_type:
            reduction type for the encoder
        :param for_context:
            whether this is the context encoder (as opposed to the candidate encoder).
            Useful for subclasses.
        :return:
            a TransformerEncoder, initialized correctly
        """
        n_positions = get_n_positions_from_options(opt)
        embeddings = self._get_embeddings(dict_=dict_, null_idx=null_idx, embedding_size=opt['embedding_size'])
        return TransformerEncoder(n_heads=opt['n_heads'], n_layers=opt['n_layers'], embedding_size=opt['embedding_size'], ffn_size=opt['ffn_size'], vocabulary_size=len(dict_), embedding=embeddings, dropout=opt['dropout'], attention_dropout=opt['attention_dropout'], relu_dropout=opt['relu_dropout'], padding_idx=null_idx, learn_positional_embeddings=opt['learn_positional_embeddings'], embeddings_scale=opt['embeddings_scale'], reduction_type=reduction_type, n_positions=n_positions, n_segments=opt.get('n_segments', 2), activation=opt['activation'], variant=opt['variant'], output_scaling=opt['output_scaling'])

    def _get_embeddings(self, dict_, null_idx, embedding_size):
        embeddings = torch.nn.Embedding(len(dict_), embedding_size, padding_idx=null_idx)
        torch.nn.init.normal_(embeddings.weight, 0, embedding_size ** (-0.5))
        return embeddings

    def attend(self, attention_layer, queries, keys, values, mask):
        """
        Apply attention.

        :param attention_layer:
            nn.Module attention layer to use for the attention
        :param queries:
            the queries for attention
        :param keys:
            the keys for attention
        :param values:
            the values for attention
        :param mask:
            mask for the attention keys

        :return:
            the result of applying attention to the values, with weights computed
            wrt to the queries and keys.
        """
        if keys is None:
            keys = values
        if isinstance(attention_layer, PolyBasicAttention):
            return attention_layer(queries, keys, mask_ys=mask, values=values)
        elif isinstance(attention_layer, MultiHeadAttention):
            return attention_layer(queries, keys, values, mask)
        else:
            raise Exception('Unrecognized type of attention')

    def encode(self, cand_tokens: Optional[torch.Tensor], **ctxt_inputs: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode a text sequence.

        :param ctxt_inputs:
            Dictionary of context inputs. If not empty, should contain at least
            'ctxt_tokens', a 2D long tensor of shape batchsize x sent_len
        :param cand_tokens:
            3D long tensor, batchsize x num_cands x sent_len
            Note this will actually view it as a 2D tensor
        :return:
            (ctxt_rep, ctxt_mask, cand_rep)
            - ctxt_rep 3D float tensor, batchsize x n_codes x dim
            - ctxt_mask byte:  batchsize x n_codes (all 1 in case
            of polyencoder with code. Which are the vectors to use
            in the ctxt_rep)
            - cand_rep (3D float tensor) batchsize x num_cands x dim
        """
        cand_embed = None
        ctxt_rep = None
        ctxt_rep_mask = None
        if cand_tokens is not None:
            assert len(cand_tokens.shape) == 3
            bsz = cand_tokens.size(0)
            num_cands = cand_tokens.size(1)
            cand_embed = self.encoder_cand(cand_tokens.view(bsz * num_cands, -1))
            cand_embed = cand_embed.view(bsz, num_cands, -1)
        if len(ctxt_inputs) > 0:
            assert 'ctxt_tokens' in ctxt_inputs
            if ctxt_inputs['ctxt_tokens'] is not None:
                assert len(ctxt_inputs['ctxt_tokens'].shape) == 2
            bsz = self._get_context_batch_size(**ctxt_inputs)
            ctxt_out, ctxt_mask = self.encoder_ctxt(**self._context_encoder_input(ctxt_inputs))
            dim = ctxt_out.size(2)
            if self.type == 'codes':
                ctxt_rep = self.attend(self.code_attention, queries=self.codes.repeat(bsz, 1, 1), keys=ctxt_out, values=ctxt_out, mask=ctxt_mask)
                ctxt_rep_mask = ctxt_rep.new_ones(bsz, self.n_codes).byte()
            elif self.type == 'n_first':
                if ctxt_out.size(1) < self.n_codes:
                    difference = self.n_codes - ctxt_out.size(1)
                    extra_rep = ctxt_out.new_zeros(bsz, difference, dim)
                    ctxt_rep = torch.cat([ctxt_out, extra_rep], dim=1)
                    extra_mask = ctxt_mask.new_zeros(bsz, difference)
                    ctxt_rep_mask = torch.cat([ctxt_mask, extra_mask], dim=1)
                else:
                    ctxt_rep = ctxt_out[:, 0:self.n_codes, :]
                    ctxt_rep_mask = ctxt_mask[:, 0:self.n_codes]
        return (ctxt_rep, ctxt_rep_mask, cand_embed)

    def _get_context_batch_size(self, **ctxt_inputs: torch.Tensor) -> int:
        """
        Return the batch size of the context.

        Can be overridden by subclasses that do not always have text tokens in the
        context.
        """
        return ctxt_inputs['ctxt_tokens'].size(0)

    def _context_encoder_input(self, ctxt_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return the inputs to the context encoder as a dictionary.

        Must return a dictionary.  This will be passed directly into the model via
        `**kwargs`, i.e.,

        >>> encoder_ctxt(**_context_encoder_input(ctxt_inputs))

        This is needed because the context encoder's forward function may have different
        argument names than that of the model itself. This is intentionally overridable
        so that richer models can pass additional inputs.
        """
        assert set(ctxt_inputs.keys()) == {'ctxt_tokens'}
        return {'input': ctxt_inputs['ctxt_tokens']}

    def score(self, ctxt_rep, ctxt_rep_mask, cand_embed):
        """
        Score the candidates.

        :param ctxt_rep:
            3D float tensor, bsz x ctxt_len x dim
        :param ctxt_rep_mask:
            2D byte tensor, bsz x ctxt_len, in case there are some elements
            of the ctxt that we should not take into account.
        :param cand_embed: 3D float tensor, bsz x num_cands x dim

        :return: scores, 2D float tensor: bsz x num_cands
        """
        ctxt_final_rep = self.attend(self.attention, cand_embed, ctxt_rep, ctxt_rep, ctxt_rep_mask)
        scores = torch.sum(ctxt_final_rep * cand_embed, 2)
        return scores

    def forward(self, cand_tokens=None, ctxt_rep=None, ctxt_rep_mask=None, cand_rep=None, **ctxt_inputs):
        """
        Forward pass of the model.

        Due to a limitation of parlai, we have to have one single model
        in the agent. And because we want to be able to use data-parallel,
        we need to have one single forward() method.
        Therefore the operation_type can be either 'encode' or 'score'.

        :param ctxt_inputs:
            Dictionary of context inputs. Will include at least 'ctxt_tokens',
            containing tokenized contexts
        :param cand_tokens:
            tokenized candidates
        :param ctxt_rep:
            (bsz x num_codes x hsz)
            encoded representation of the context. If self.type == 'codes', these
            are the context codes. Otherwise, they are the outputs from the
            encoder
        :param ctxt_rep_mask:
            mask for ctxt rep
        :param cand_rep:
            encoded representation of the candidates
        """
        if len(ctxt_inputs) > 0 or cand_tokens is not None:
            return self.encode(cand_tokens=cand_tokens, **ctxt_inputs)
        elif ctxt_rep is not None and ctxt_rep_mask is not None and (cand_rep is not None):
            return self.score(ctxt_rep, ctxt_rep_mask, cand_rep)
        raise Exception('Unsupported operation')