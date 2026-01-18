import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.feature_maps import (
@register_attention('favor', FavorAttentionConfig)
class FavorAttention(Attention):

    def __init__(self, causal: bool=False, dropout: float=0.0, dim_features: Optional[int]=None, dim_head: Optional[int]=None, iter_before_redraw: Optional[int]=None, feature_map_type: FeatureMapType=FeatureMapType.SMReg, normalize_inputs: bool=False, *_, **__):
        """
        Kernelized attention, as proposed in Performers_
        ("Rethinking attention with performers." K. Choromanski et al. (2020).).

        FAVOR stands for "Fast Attention Via positive Orthogonal Random features"

        Args:
            dropout (float): the probability of an output to be randomly dropped at training time
            dim_features (int): the dimension of the random features space
            iter_before_redraw (int): the number of steps (forward calls) before a redraw of the features
            feature_map_type (FeatureMapType): the type of feature map being used,
            for instance orthogonal random features.

        .. _Performers: https://arxiv.org/pdf/2009.14794v1.pdf
        """
        super().__init__()
        self.causal = causal
        self.iter_before_redraw = 2 * iter_before_redraw if iter_before_redraw is not None else iter_before_redraw
        self.normalize_inputs = normalize_inputs
        self.feature_map_type = feature_map_type
        self.attn_drop = nn.Dropout(dropout, inplace=True)
        if dim_features is None:
            assert dim_head is not None, 'dim_features or dim_head needs to be passed'
            self.dim_features = math.ceil(dim_head * (1 + math.log2(dim_head)))
            self.dim_features = 2 * (self.dim_features // 2)
            logger.info(f'FAVOR: Automatically setting the random mapping dimension to {self.dim_features} from {dim_head}')
        else:
            self.dim_features = dim_features
        feature_map_constructor = {FeatureMapType.SMHyp: SMHyperbolic, FeatureMapType.SMReg: SMReg, FeatureMapType.SMOrf: SMOrf}[self.feature_map_type]
        feature_settings = {'dim_features': self.dim_features, 'iter_before_redraw': self.iter_before_redraw, 'normalize_inputs': self.normalize_inputs}
        self.feature_map: FeatureMap = feature_map_constructor(**feature_settings)
        self.supports_attention_mask = False
        self.supports_key_padding_mask = False

    @staticmethod
    def _maybe_promote(x: torch.Tensor) -> torch.Tensor:
        return x.float() if x.dtype == torch.float16 else x

    @staticmethod
    def _causal_attention(k_prime: torch.Tensor, q_prime: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ref_v = torch.ones_like(v.unsqueeze(2))
        Gps = k_prime.unsqueeze(3) * v.unsqueeze(2)
        Grenorm = k_prime.unsqueeze(3) * ref_v
        att_raw = torch.einsum('bcfe,bcf->bce', Gps, q_prime)
        att_norm = torch.einsum('bcfe,bcf->bce', Grenorm, q_prime)
        att_raw = att_raw.cumsum(2)
        att_norm = att_norm.cumsum(2)
        return (att_raw, att_norm)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *_, **__):
        k_prime = self.feature_map(k)
        q_prime = self.feature_map(q)
        with autocast(enabled=False):
            k_prime = self._maybe_promote(k_prime)
            q_prime = self._maybe_promote(q_prime)
            v = self._maybe_promote(v)
            if not self.causal:
                att_normalization = q_prime @ (k_prime.transpose(-2, -1) @ torch.ones_like(v))
                att_raw = q_prime @ (k_prime.transpose(-2, -1) @ v)
            else:
                att_raw, att_normalization = self._causal_attention(k_prime, q_prime, v)
            att = att_raw / att_normalization
        if self.attn_drop is not None:
            att = self.attn_drop(att)
        return att