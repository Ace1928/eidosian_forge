import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class SquimObjective(nn.Module):
    """Speech Quality and Intelligibility Measures (SQUIM) model that predicts **objective** metric scores
    for speech enhancement (e.g., STOI, PESQ, and SI-SDR).

    Args:
        encoder (torch.nn.Module): Encoder module to transform 1D waveform to 2D feature representation.
        dprnn (torch.nn.Module): DPRNN module to model sequential feature.
        branches (torch.nn.ModuleList): Transformer branches in which each branch estimate one objective metirc score.
    """

    def __init__(self, encoder: nn.Module, dprnn: nn.Module, branches: nn.ModuleList):
        super(SquimObjective, self).__init__()
        self.encoder = encoder
        self.dprnn = dprnn
        self.branches = branches

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input waveforms. Tensor with dimensions `(batch, time)`.

        Returns:
            List(torch.Tensor): List of score Tenosrs. Each Tensor is with dimension `(batch,)`.
        """
        if x.ndim != 2:
            raise ValueError(f'The input must be a 2D Tensor. Found dimension {x.ndim}.')
        x = x / (torch.mean(x ** 2, dim=1, keepdim=True) ** 0.5 * 20)
        out = self.encoder(x)
        out = self.dprnn(out)
        scores = []
        for branch in self.branches:
            scores.append(branch(out).squeeze(dim=1))
        return scores