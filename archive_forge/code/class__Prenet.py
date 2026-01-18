import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
class _Prenet(nn.Module):
    """Prenet Module. It is consists of ``len(output_size)`` linear layers.

    Args:
        in_dim (int): The size of each input sample.
        output_sizes (list): The output dimension of each linear layers.
    """

    def __init__(self, in_dim: int, out_sizes: List[int]) -> None:
        super().__init__()
        in_sizes = [in_dim] + out_sizes[:-1]
        self.layers = nn.ModuleList([_get_linear_layer(in_size, out_size, bias=False) for in_size, out_size in zip(in_sizes, out_sizes)])

    def forward(self, x: Tensor) -> Tensor:
        """Pass the input through Prenet.

        Args:
            x (Tensor): The input sequence to Prenet with shape (n_batch, in_dim).

        Return:
            x (Tensor): Tensor with shape (n_batch, sizes[-1])
        """
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x