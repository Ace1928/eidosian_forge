from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):

    @override
    def _check_input_dim(self, input: Tensor) -> None:
        return