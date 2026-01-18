import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
class _BLSTM(torch.nn.Module):
    """
    BiLSTM with same hidden units as input dim.
    If `max_steps` is not None, input will be splitting in overlapping
    chunks and the LSTM applied separately on each chunk.
    Args:
        dim (int): dimensions at LSTM layer.
        layers (int, optional): number of LSTM layers. (default: 1)
        skip (bool, optional): (default: ``False``)
    """

    def __init__(self, dim, layers: int=1, skip: bool=False):
        super().__init__()
        self.max_steps = 200
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """BLSTM forward call

        Args:
            x (torch.Tensor): input tensor for BLSTM shape is `(batch_size, dim, time_steps)`

        Returns:
            Tensor
                Output after being run through bidirectional LSTM. Shape is `(batch_size, dim, time_steps)`
        """
        B, C, T = x.shape
        y = x
        framed = False
        width = 0
        stride = 0
        nframes = 0
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = _unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out
        if self.skip:
            x = x + y
        return x