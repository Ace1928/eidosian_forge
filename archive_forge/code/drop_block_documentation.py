import torch
import torch.fx
import torch.nn.functional as F
from torch import nn, Tensor
from ..utils import _log_api_usage_once

        Args:
            input (Tensor): Input feature map on which some areas will be randomly
                dropped.
        Returns:
            Tensor: The tensor after DropBlock layer.
        