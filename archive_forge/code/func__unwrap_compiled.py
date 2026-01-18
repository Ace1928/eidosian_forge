import inspect
from copy import deepcopy
from functools import wraps
from typing import (
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch import nn as nn
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override
from lightning_fabric.plugins import Precision
from lightning_fabric.strategies import Strategy
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.data import _set_sampler_epoch
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.types import Optimizable
def _unwrap_compiled(obj: Union[Any, 'OptimizedModule']) -> Tuple[Union[Any, nn.Module], Optional[Dict[str, Any]]]:
    """Removes the :class:`torch._dynamo.OptimizedModule` around the object if it is wrapped.

    Use this function before instance checks against e.g. :class:`_FabricModule`.

    """
    if not _TORCH_GREATER_EQUAL_2_0:
        return (obj, None)
    from torch._dynamo import OptimizedModule
    if isinstance(obj, OptimizedModule):
        if (compile_kwargs := getattr(obj, '_compile_kwargs', None)) is None:
            raise RuntimeError('Failed to determine the arguments that were used to compile the module. Make sure to import lightning before `torch.compile` is used.')
        return (obj._orig_mod, compile_kwargs)
    return (obj, None)