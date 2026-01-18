import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
LayerScale forward call

        Args:
            x (torch.Tensor): input tensor for LayerScale

        Returns:
            Tensor
                Output after rescaling tensor.
        