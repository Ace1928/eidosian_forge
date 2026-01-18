from typing import Any, ContextManager, Literal
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
Plugin for training with double (``torch.float64``) precision.