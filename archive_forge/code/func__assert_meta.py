from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode
def _assert_meta(grad, size, stride, dtype):
    assert grad.size() == size, 'size mismatch'
    assert grad.stride() == stride, 'stride mismatch'
    assert grad.dtype == dtype, 'dtype mismatch'
    return grad