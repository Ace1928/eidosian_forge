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
def _unwrap_objects(collection: Any) -> Any:

    def _unwrap(obj: Union[_FabricModule, _FabricOptimizer, _FabricDataLoader]) -> Union[nn.Module, Optimizer, DataLoader]:
        if isinstance((unwrapped := _unwrap_compiled(obj)[0]), _FabricModule):
            return _unwrap_compiled(unwrapped._forward_module)[0]
        if isinstance(obj, _FabricOptimizer):
            return obj.optimizer
        if isinstance(obj, _FabricDataLoader):
            return obj._dataloader
        return obj
    types = [_FabricModule, _FabricOptimizer, _FabricDataLoader]
    if _TORCH_GREATER_EQUAL_2_0:
        from torch._dynamo import OptimizedModule
        types.append(OptimizedModule)
    return apply_to_collection(collection, dtype=tuple(types), function=_unwrap)