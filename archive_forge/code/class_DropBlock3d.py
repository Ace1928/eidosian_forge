import torch
import torch.fx
import torch.nn.functional as F
from torch import nn, Tensor
from ..utils import _log_api_usage_once
class DropBlock3d(DropBlock2d):
    """
    See :func:`drop_block3d`.
    """

    def __init__(self, p: float, block_size: int, inplace: bool=False, eps: float=1e-06) -> None:
        super().__init__(p, block_size, inplace, eps)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input feature map on which some areas will be randomly
                dropped.
        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        return drop_block3d(input, self.p, self.block_size, self.inplace, self.eps, self.training)