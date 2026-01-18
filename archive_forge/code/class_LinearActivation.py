import math
import warnings
from collections import OrderedDict
import torch
from packaging import version
from torch import Tensor, nn
from .utils import logging
class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input