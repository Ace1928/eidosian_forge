from typing import Any, ContextManager, Literal
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
class DoublePrecision(Precision):
    """Plugin for training with double (``torch.float64``) precision."""
    precision: Literal['64-true'] = '64-true'

    @override
    def convert_module(self, module: Module) -> Module:
        return module.double()

    @override
    def tensor_init_context(self) -> ContextManager:
        return _DtypeContextManager(torch.double)

    @override
    def module_init_context(self) -> ContextManager:
        return self.tensor_init_context()

    @override
    def forward_context(self) -> ContextManager:
        return self.tensor_init_context()

    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.double)

    @override
    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())