from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
from torchaudio.models import Emformer
class _TimeReduction(torch.nn.Module):
    """Coalesces frames along time dimension into a
    fewer number of frames with higher feature dimensionality.

    Args:
        stride (int): number of frames to merge for each output frame.
    """

    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        B: batch size;
        T: maximum input sequence length in batch;
        D: feature dimension of each input sequence frame.

        Args:
            input (torch.Tensor): input sequences, with shape `(B, T, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor):
                torch.Tensor
                    output sequences, with shape
                    `(B, T  // stride, D * stride)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output sequences.
        """
        B, T, D = input.shape
        num_frames = T - T % self.stride
        input = input[:, :num_frames, :]
        lengths = lengths.div(self.stride, rounding_mode='trunc')
        T_max = num_frames // self.stride
        output = input.reshape(B, T_max, D * self.stride)
        output = output.contiguous()
        return (output, lengths)