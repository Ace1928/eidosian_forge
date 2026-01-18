import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
@dataclasses.dataclass(frozen=True)
@compatibility(is_backward_compatible=False)
class OrtBackendOptions:
    """Options for constructing an ``OrtBackend``, the ONNX Runtime
    backend (``"onnxrt"``) for ``torch.compile``.

    Example::

        >>> @torch.compile(
        ...     backend="onnxrt",
        ...     options=torch.onnx._OrtBackendOptions(...),
        ... )
        ... def ort_function(x):
        ...     return x ** x
    """
    preferred_execution_providers: Optional[Sequence[OrtExecutionProvider]] = None
    'An optional sequence of execution providers to be prioritized ahead of any\n    execution providers that may be inferred (see ``infer_execution_providers``).\n    '
    infer_execution_providers: bool = True
    'Whether to infer an execution provider from ``torch.device`` bound to inputs or found in the graph.'
    default_execution_providers: Optional[Sequence[OrtExecutionProvider]] = None
    'The default fallback execution providers. If not specified, one will be\n    be selected based on the host environment (most likely ``"CPUExecutionProvider"``).\n    '
    preallocate_output: bool = False
    "If ``True``, allocate memory for ONNX Runtime's outputs on the PyTorch side."
    use_aot_autograd: bool = True
    "Whether to wrap the ``OrtBackend`` with TorchDynamo's aot_autograd backend\n    to support training (i.e., backward graphs are also sent to ``OrtBackend``).\n\n    Symbolic execution is used to capture the forward pass and backward passes as a single graph.\n    Then, a selected graph partition algorithm (``min_cut_rematerialization_partition``) is used\n    to split the entire graph into forward sub-graph and backward sub-graph. Finally, both\n    sub-graphs are compiled by ``OrtBackend``.\n    "
    export_options: Optional['torch.onnx.ExportOptions'] = None
    'Options for the TorchDynamo-based ONNX exporter used by the ``OrtBackend``.'
    ort_session_options: Optional['onnxruntime.SessionOptions'] = None
    'Options for the ``onnxruntime.InferenceSession`` used by the ``OrtBackend``.'