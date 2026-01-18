from typing import Any, Dict
import torch
from parlai.agents.image_seq2seq.modules import ContextWithImageEncoder
from parlai.agents.transformer.modules import get_n_positions_from_options
from parlai.agents.transformer.polyencoder import PolyencoderAgent, PolyEncoderModule
from parlai.core.torch_agent import Batch
from parlai.core.torch_image_agent import TorchImageAgent
from parlai.utils.misc import warn_once
class ImagePolyencoderModule(PolyEncoderModule):
    """
    Poly-encoder model with image features.

    Model that allows encoding image features and adding or concatenating them to the
    context encoding.
    """

    def get_encoder(self, opt, dict_, null_idx, reduction_type, for_context: bool):
        """
        Return encoder that allows for image features to be passed in, given options.

        :param opt:
            opt dict
        :param dict:
            dictionary agent
        :param null_idx:
            null/pad index into dict
        :param reduction_type: only used for compatibility with the superclass method
        :param for_context:
            whether this is the context encoder (as opposed to the candidate encoder)
        :return:
            either a TransformerEncoder or a ContextWithImageEncoder, initialized
            correctly
        """
        if for_context:
            if reduction_type is not None:
                raise NotImplementedError('No encoder output reductions supported!')
            n_positions = get_n_positions_from_options(opt)
            embeddings = self._get_embeddings(dict_=dict_, null_idx=null_idx, embedding_size=opt['embedding_size'])
            return ContextWithImageEncoder(n_heads=opt['n_heads'], n_layers=opt['n_layers'], embedding_size=opt['embedding_size'], ffn_size=opt['ffn_size'], vocabulary_size=len(dict_), embedding=embeddings, dropout=opt['dropout'], attention_dropout=opt['attention_dropout'], relu_dropout=opt['relu_dropout'], padding_idx=null_idx, learn_positional_embeddings=opt['learn_positional_embeddings'], embeddings_scale=opt['embeddings_scale'], n_positions=n_positions, n_segments=opt['n_segments'], activation=opt['activation'], variant=opt['variant'], output_scaling=opt['output_scaling'], image_encoder_num_layers=opt['image_encoder_num_layers'], image_features_dim=opt['image_features_dim'], image_combination_mode=opt['image_combination_mode'], n_image_tokens=opt['n_image_tokens'])
        else:
            return super().get_encoder(opt=opt, dict_=dict_, null_idx=null_idx, reduction_type=reduction_type, for_context=for_context)

    def _context_encoder_input(self, ctxt_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override PolyEncoderModule's inputs into the context encoder.
        """
        assert set(ctxt_inputs.keys()) == {'ctxt_tokens', 'ctxt_image'}
        return {'src_tokens': ctxt_inputs['ctxt_tokens'], 'image_features': ctxt_inputs['ctxt_image']}

    def _get_context_batch_size(self, **ctxt_inputs: torch.Tensor) -> int:
        """
        Return the batch size of the context.
        """
        if ctxt_inputs['ctxt_tokens'] is not None:
            return ctxt_inputs['ctxt_tokens'].size(0)
        else:
            return ctxt_inputs['ctxt_image'].size(0)