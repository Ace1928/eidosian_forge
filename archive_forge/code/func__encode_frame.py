import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_encodec import EncodecConfig
def _encode_frame(self, input_values: torch.Tensor, bandwidth: float, padding_mask: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
        Encodes the given input using the underlying VQVAE. If `config.normalize` is set to `True` the input is first
        normalized. The padding mask is required to compute the correct scale.
        """
    length = input_values.shape[-1]
    duration = length / self.config.sampling_rate
    if self.config.chunk_length_s is not None and duration > 1e-05 + self.config.chunk_length_s:
        raise RuntimeError(f'Duration of frame ({duration}) is longer than chunk {self.config.chunk_length_s}')
    scale = None
    if self.config.normalize:
        input_values = input_values * padding_mask
        mono = torch.sum(input_values, 1, keepdim=True) / input_values.shape[1]
        scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-08
        input_values = input_values / scale
    embeddings = self.encoder(input_values)
    codes = self.quantizer.encode(embeddings, bandwidth)
    codes = codes.transpose(0, 1)
    return (codes, scale)