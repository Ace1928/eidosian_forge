from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple
import torch.nn.functional as F
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation
from ..utils import _log_api_usage_once
class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN.

    Args:
        results (List[Tensor]): the result of the FPN
        x (List[Tensor]): the original feature maps
        names (List[str]): the names for each one of the
            original feature maps

    Returns:
        results (List[Tensor]): the extended set of results
            of the FPN
        names (List[str]): the extended set of names for the results
    """

    def forward(self, results: List[Tensor], x: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        pass