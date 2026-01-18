from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode
@_trace_wrapped_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _trace_wrapped_op_dense(*args, fn):
    mode = _get_current_dispatch_mode()
    assert mode is None, 'Mode should never be enabled for CPU/CUDA key'
    return fn(*args)