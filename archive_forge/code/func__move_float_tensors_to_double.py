from contextlib import contextmanager
from typing import Any, ContextManager, Generator, Literal
import torch
import torch.nn as nn
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from pytorch_lightning.plugins.precision.precision import Precision
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation
@staticmethod
def _move_float_tensors_to_double(collection: Any) -> Any:
    return apply_to_collection(collection, Tensor, function=_convert_fp_tensor, dst_type=torch.double)