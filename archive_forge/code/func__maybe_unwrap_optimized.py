from typing import Union
import torch
import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_1
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, SingleDeviceStrategy, Strategy
from pytorch_lightning.utilities.model_helpers import _check_mixed_imports
def _maybe_unwrap_optimized(model: object) -> 'pl.LightningModule':
    if not _TORCH_GREATER_EQUAL_2_0:
        if not isinstance(model, pl.LightningModule):
            _check_mixed_imports(model)
            raise TypeError(f'`model` must be a `LightningModule`, got `{type(model).__qualname__}`')
        return model
    from torch._dynamo import OptimizedModule
    if isinstance(model, OptimizedModule):
        return from_compiled(model)
    if isinstance(model, pl.LightningModule):
        return model
    _check_mixed_imports(model)
    raise TypeError(f'`model` must be a `LightningModule` or `torch._dynamo.OptimizedModule`, got `{type(model).__qualname__}`')