import logging
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter
from .wavlm_attention import WavLMSelfAttention
class FeatureProjection(Module):
    """Layer that connects FeatureExtractor and Encoder

    Projects features to encoder dimension.

    Args:
        in_features (int): Input feature dim.
        out_features (int): Output feature dim.
        dropout (float): Dropout probability.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor):
                Feature Tensor. shape: ``[batch, frame, in_feature]``
        Returns:
            Tensor: Projected features. ``[batch, frame, out_feature]``.
        """
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x