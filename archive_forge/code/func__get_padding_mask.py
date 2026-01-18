import logging
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter
from .wavlm_attention import WavLMSelfAttention
def _get_padding_mask(input: Tensor, lengths: Tensor) -> Tensor:
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        input (Tensor): The padded Tensor of dimension `[batch, max_len, frequency]`.
        lengths (Tensor): The lengths Tensor of dimension `[batch,]`.

    Returns:
        (Tensor): The padding mask.
    """
    batch_size, max_len, _ = input.shape
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
    return mask