import contextlib
import functools
import itertools
import logging
from typing import Dict, List, Optional
import torch._C
import torch.fx
import torch.nn
import torch.onnx.operators
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import deepcopy_to_fake_tensor, get_fake_value, get_real_value
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builtin import BuiltinVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.tensor import SymNodeVariable
from torch._guards import Source
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree
from ..exc import (
from ..source import FSDPNNModuleSource, GetItemSource, NNModuleSource
from ..utils import proxy_args_kwargs
from .dicts import ConstDictVariable
from .lists import ListVariable, TupleVariable
from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable
class TorchHigherOrderOperatorVariable(VariableTracker):

    def __init__(self, value, source: Optional[Source]=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.source = source

    @staticmethod
    def make(value, source=None, **kwargs):
        if value.__name__ == 'cond':
            return CondHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in ('map', 'map_impl'):
            return MapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == 'executorch_call_delegate':
            return ExecutorchCallDelegateHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == 'out_dtype':
            return OutDtypeHigherOrderVariable(value, source, **kwargs)
        elif value is torch._functorch.eager_transforms.grad_impl:
            return FunctorchGradHigherOrderVariable(value, source, **kwargs)
        elif value is torch._functorch.vmap.vmap_impl:
            return FunctorchVmapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in ('trampoline_autograd_fwd', 'trampoline_autograd_bwd', 'trampoline_autograd_apply'):
            return AutogradFunctionMethodHigherOrderVariable(value=value, source=source, **kwargs)
        elif value.__name__ == 'wrap':
            return WrapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in ('wrap_activation_checkpoint', 'tag_activation_checkpoint'):
            return CheckpointHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == '_export_tracepoint':
            return ExportTracepointHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == 'trace_wrapped':
            return TraceWrappedHigherOrderOperatorVariable(value, source, **kwargs)
        else:
            unimplemented(f'HigherOrderOperator {value.__name__}')

    def call_function(self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]) -> VariableTracker:
        unimplemented(f'HigherOrderOperator {self.value.__name__}')