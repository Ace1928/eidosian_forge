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
class LightningDoublePrecisionModule(_DeviceDtypeModuleMixin, nn.Module):
    """LightningModule wrapper which converts incoming floating point data in ``*_step`` and ``forward`` to double
    (``torch.float64``) precision.

    .. deprecated:: Use :meth:`~pytorch_lightning.core.hooks.ModelHooks.configure_model` instead.

    Args:
        pl_module: the model to wrap

    """

    def __init__(self, pl_module: 'pl.LightningModule') -> None:
        super().__init__()
        rank_zero_deprecation(f'The `{type(self).__name__}` is deprecated and no longer needed. Convert the inputs to the `*_step` methods directly using `trainer.precision_plugin.convert_input(...)`.')
        self.module = pl_module
        _ddp_params_and_buffers_to_ignore = getattr(pl_module, '_ddp_params_and_buffers_to_ignore', [])
        self._ddp_params_and_buffers_to_ignore = [f'module.{p}' for p in _ddp_params_and_buffers_to_ignore]

    @staticmethod
    def _move_float_tensors_to_double(collection: Any) -> Any:
        return apply_to_collection(collection, Tensor, function=_convert_fp_tensor, dst_type=torch.double)

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.training_step(*LightningDoublePrecisionModule._move_float_tensors_to_double(args), **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs))

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.validation_step(*LightningDoublePrecisionModule._move_float_tensors_to_double(args), **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs))

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.test_step(*LightningDoublePrecisionModule._move_float_tensors_to_double(args), **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs))

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.predict_step(*LightningDoublePrecisionModule._move_float_tensors_to_double(args), **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs))

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*LightningDoublePrecisionModule._move_float_tensors_to_double(args), **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs))