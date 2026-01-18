import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
class _LocationLayer(nn.Module):
    """Location layer used in the Attention model.

    Args:
        attention_n_filter (int): Number of filters for attention model.
        attention_kernel_size (int): Kernel size for attention model.
        attention_hidden_dim (int): Dimension of attention hidden representation.
    """

    def __init__(self, attention_n_filter: int, attention_kernel_size: int, attention_hidden_dim: int):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = _get_conv1d_layer(2, attention_n_filter, kernel_size=attention_kernel_size, padding=padding, bias=False, stride=1, dilation=1)
        self.location_dense = _get_linear_layer(attention_n_filter, attention_hidden_dim, bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat: Tensor) -> Tensor:
        """Location layer used in the Attention model.

        Args:
            attention_weights_cat (Tensor): Cumulative and previous attention weights
                with shape (n_batch, 2, max of ``text_lengths``).

        Returns:
            processed_attention (Tensor): Cumulative and previous attention weights
                with shape (n_batch, ``attention_hidden_dim``).
        """
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention