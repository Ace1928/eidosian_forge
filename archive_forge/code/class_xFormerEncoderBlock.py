import logging
from dataclasses import asdict
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers._deprecation_warning import deprecated_function
from xformers.components import (
from xformers.components.attention import AttentionMask
from xformers.components.feedforward import build_feedforward
from xformers.components.positional_embedding import build_positional_embedding
from xformers.components.residual import get_deepnorm_coefficients
from xformers.components.simplicial_embedding import SimplicialEmbedding
from xformers.factory.block_configs import (
class xFormerEncoderBlock(torch.nn.Module):
    """A vanilla Transformer Encoder block"""

    def __init__(self, config: xFormerEncoderConfig, **kwargs):
        super().__init__()
        deprecated_function(self)
        self.reversible_f = None
        self.reversible_g = None
        self.residual_norm_style = config.residual_norm_style
        self.dim_model = config.dim_model
        if config.position_encoding_config is not None and config.layer_position.is_first():
            self.pose_encoding = build_positional_embedding(asdict(config.position_encoding_config))
            pos_encoding_dim = config.position_encoding_config.dim_model
            mha_dim = config.multi_head_config['dim_model']
            if pos_encoding_dim != mha_dim:
                logger.warning(f'The embedding dim and model dim do not match ({pos_encoding_dim} vs {mha_dim}), adding a projector layer.')
                self.embedding_projector = nn.Linear(pos_encoding_dim, mha_dim)
        else:
            self.pose_encoding = None
        if config.residual_norm_style == ResidualNormStyle.DeepNorm:
            deep_norm_coefficients, _ = get_deepnorm_coefficients(encoder_layers=config.num_layers, decoder_layers=0)
            assert deep_norm_coefficients is not None
            residual_scale = deep_norm_coefficients.alpha
        else:
            residual_scale = 1.0
        ln_factory = _get_ln_factory(config.dim_model, config.residual_norm_style, use_triton=config.use_triton, residual=True, residual_scale=residual_scale, normalization=config.normalization)
        mha = build_multi_head_attention(config.multi_head_config)
        feedforward = build_feedforward(asdict(config.feedforward_config))
        self.supports_attention_mask = mha.attention.supports_attention_mask
        self.requires_same_k_q_dimensions = mha.attention.requires_same_k_q_dimensions
        self.causal = mha.attention.causal if hasattr(mha.attention, 'causal') else False
        self.wrap_att = ln_factory(mha)
        self.wrap_ff: Union[Residual, PostNorm] = ln_factory(feedforward)
        if config.residual_norm_style == ResidualNormStyle.Pre and config.layer_position.is_last():
            self.wrap_ff = PostNorm(config.dim_model, self.wrap_ff, normalization=config.normalization, use_triton=config.use_triton)
        self.simplicial_embedding: Optional[SimplicialEmbedding] = None
        if config.simplicial_embeddings is not None and config.layer_position.is_last():
            self.simplicial_embedding = SimplicialEmbedding(**config.simplicial_embeddings)
        self.patch_emb: Optional[nn.Module] = None
        if config.patch_embedding_config is not None:
            self.patch_emb = build_patch_embedding(PatchEmbeddingConfig(**config.patch_embedding_config))

    @classmethod
    def from_config(cls, config: xFormerEncoderConfig):
        return cls(config)

    @staticmethod
    def get_reversible_layer(config) -> Tuple[nn.Module, nn.Module]:
        ln_factory = _get_ln_factory(config.dim_model, config.residual_norm_style, residual=False, use_triton=config.use_triton, normalization=config.normalization)
        mha = build_multi_head_attention(config.multi_head_config)
        feedforward = build_feedforward(asdict(config.feedforward_config))
        reversible_f = ln_factory(mha)
        reversible_g = ln_factory(feedforward)
        return (reversible_f, reversible_g)

    def forward(self, x: torch.Tensor, att_mask: Optional[Union[torch.Tensor, AttentionMask]]=None, input_mask: Optional[torch.Tensor]=None):
        if self.patch_emb is not None:
            x = self.patch_emb(x)
        if self.pose_encoding is not None:
            x = self.pose_encoding(x)
            if hasattr(self, 'embedding_projector'):
                x = self.embedding_projector(x)
        if input_mask is not None:
            q = x
            k = x * input_mask.unsqueeze(-1)
            v = k
        else:
            q, k, v = (x, x, x)
        x = self.wrap_att(inputs=[q, k, v], att_mask=att_mask)
        x = self.wrap_ff(inputs=[x])
        if self.simplicial_embedding is not None:
            x = self.simplicial_embedding(x)
        return x