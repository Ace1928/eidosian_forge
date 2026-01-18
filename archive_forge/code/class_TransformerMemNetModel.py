import math
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf, PipelineHelper
class TransformerMemNetModel(nn.Module):
    """
    Model which takes context, memories, candidates and encodes them.
    """

    @classmethod
    def build_encoder(cls, opt, dictionary, embedding=None, padding_idx=None, reduction_type='mean', n_positions=1024, n_segments=0):
        n_layers = opt['n_encoder_layers'] if opt.get('n_encoder_layers', -1) > 0 else opt['n_layers']
        return TransformerEncoder(n_heads=opt['n_heads'], n_layers=n_layers, embedding_size=opt['embedding_size'], ffn_size=opt['ffn_size'], vocabulary_size=len(dictionary), embedding=embedding, dropout=opt['dropout'], attention_dropout=opt['attention_dropout'], relu_dropout=opt['relu_dropout'], padding_idx=padding_idx, learn_positional_embeddings=opt['learn_positional_embeddings'], embeddings_scale=opt['embeddings_scale'], reduction_type=reduction_type, n_positions=n_positions, n_segments=n_segments, activation=opt['activation'], variant=opt['variant'], output_scaling=opt['output_scaling'])

    def __init__(self, opt, dictionary):
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[dictionary.null_token]
        self.embeddings = _create_embeddings(dictionary, opt['embedding_size'], self.pad_idx)
        self.share_word_embedding = opt.get('share_word_embeddings', True)
        if not self.share_word_embedding:
            self.cand_embeddings = _create_embeddings(dictionary, opt['embedding_size'], self.pad_idx)
        if not opt.get('learn_embeddings'):
            self.embeddings.weight.requires_grad = False
            if not self.share_word_embedding:
                self.cand_embeddings.weight.requires_grad = False
        n_positions = get_n_positions_from_options(opt)
        if n_positions < 0:
            raise ValueError('n_positions must be positive')
        self.reduction_type = opt.get('reduction_type', 'mean')
        self.n_segments = opt.get('n_segments', 0)
        self.context_encoder = self.build_encoder(opt, dictionary, self.embeddings, self.pad_idx, reduction_type=self.reduction_type, n_positions=n_positions, n_segments=self.n_segments)
        if opt.get('share_encoders'):
            self.cand_encoder = TransformerResponseWrapper(self.context_encoder, self.context_encoder.out_dim)
        else:
            if not self.share_word_embedding:
                cand_embeddings = self.cand_embeddings
            else:
                cand_embeddings = self.embeddings
            self.cand_encoder = self.build_encoder(opt, dictionary, cand_embeddings, self.pad_idx, n_positions=n_positions, reduction_type=self.reduction_type, n_segments=self.n_segments)
        if opt.get('wrap_memory_encoder', False):
            self.memory_transformer = TransformerResponseWrapper(self.context_encoder, self.context_encoder.out_dim)
        else:
            self.memory_transformer = self.context_encoder
        self.attender = BasicAttention(dim=2, attn=opt['memory_attention'], residual=True)

    def encode_cand(self, words):
        """
        Encode the candidates.
        """
        if words is None:
            return None
        if words.dim() == 3:
            oldshape = words.shape
            words = words.reshape(oldshape[0] * oldshape[1], oldshape[2])
        else:
            oldshape = None
        encoded = self.cand_encoder(words)
        if oldshape is not None:
            encoded = encoded.reshape(oldshape[0], oldshape[1], -1)
        return encoded

    def encode_context_memory(self, context_w, memories_w, context_segments=None):
        """
        Encode the context and memories.
        """
        if context_w is None:
            return (None, None)
        context_h = self.context_encoder(context_w, segments=context_segments)
        if memories_w is None:
            return ([], context_h)
        bsz = memories_w.size(0)
        memories_w = memories_w.view(-1, memories_w.size(-1))
        memories_h = self.memory_transformer(memories_w)
        memories_h = memories_h.view(bsz, -1, memories_h.size(-1))
        context_h = context_h.unsqueeze(1)
        context_h, weights = self.attender(context_h, memories_h)
        return (weights, context_h)

    def forward(self, xs, mems, cands, context_segments=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] xs: input tokens IDs
        :param LongTensor[batch,num_mems,seqlen] mems: memory token IDs
        :param LongTensor[batch,num_cands,seqlen] cands: candidate token IDs
        :param LongTensor[batch,seqlen] context_segments: segment IDs for xs,
            used if n_segments is > 0 for the context encoder
        """
        weights, context_h = self.encode_context_memory(xs, mems, context_segments=context_segments)
        cands_h = self.encode_cand(cands)
        if self.opt['normalize_sent_emb']:
            context_h = context_h / context_h.norm(2, dim=1, keepdim=True)
            cands_h = cands_h / cands_h.norm(2, dim=1, keepdim=True)
        return (context_h, cands_h)