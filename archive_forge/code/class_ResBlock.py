import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
class ResBlock(nn.Module):
    """ResNet block based on *Efficient Neural Audio Synthesis* :cite:`kalchbrenner2018efficient`.

    Args:
        n_freq: the number of bins in a spectrogram. (Default: ``128``)

    Examples
        >>> resblock = ResBlock()
        >>> input = torch.rand(10, 128, 512)  # a random spectrogram
        >>> output = resblock(input)  # shape: (10, 128, 512)
    """

    def __init__(self, n_freq: int=128) -> None:
        super().__init__()
        self.resblock_model = nn.Sequential(nn.Conv1d(in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=False), nn.BatchNorm1d(n_freq), nn.ReLU(inplace=True), nn.Conv1d(in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=False), nn.BatchNorm1d(n_freq))

    def forward(self, specgram: Tensor) -> Tensor:
        """Pass the input through the ResBlock layer.
        Args:
            specgram (Tensor): the input sequence to the ResBlock layer (n_batch, n_freq, n_time).

        Return:
            Tensor shape: (n_batch, n_freq, n_time)
        """
        return self.resblock_model(specgram) + specgram