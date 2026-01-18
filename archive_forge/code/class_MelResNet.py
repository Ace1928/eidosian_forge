import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
class MelResNet(nn.Module):
    """MelResNet layer uses a stack of ResBlocks on spectrogram.

    Args:
        n_res_block: the number of ResBlock in stack. (Default: ``10``)
        n_freq: the number of bins in a spectrogram. (Default: ``128``)
        n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
        n_output: the number of output dimensions of melresnet. (Default: ``128``)
        kernel_size: the number of kernel size in the first Conv1d layer. (Default: ``5``)

    Examples
        >>> melresnet = MelResNet()
        >>> input = torch.rand(10, 128, 512)  # a random spectrogram
        >>> output = melresnet(input)  # shape: (10, 128, 508)
    """

    def __init__(self, n_res_block: int=10, n_freq: int=128, n_hidden: int=128, n_output: int=128, kernel_size: int=5) -> None:
        super().__init__()
        ResBlocks = [ResBlock(n_hidden) for _ in range(n_res_block)]
        self.melresnet_model = nn.Sequential(nn.Conv1d(in_channels=n_freq, out_channels=n_hidden, kernel_size=kernel_size, bias=False), nn.BatchNorm1d(n_hidden), nn.ReLU(inplace=True), *ResBlocks, nn.Conv1d(in_channels=n_hidden, out_channels=n_output, kernel_size=1))

    def forward(self, specgram: Tensor) -> Tensor:
        """Pass the input through the MelResNet layer.
        Args:
            specgram (Tensor): the input sequence to the MelResNet layer (n_batch, n_freq, n_time).

        Return:
            Tensor shape: (n_batch, n_output, n_time - kernel_size + 1)
        """
        return self.melresnet_model(specgram)