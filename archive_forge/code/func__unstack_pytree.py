import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._dispatch.python import suspend_functionalization
from torch._functorch.aot_autograd import AOTConfig, create_joint
from torch._functorch.eager_transforms import (
from torch._higher_order_ops.cond import (
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
def _unstack_pytree(xs):
    flat_xs, inspec = pytree.tree_flatten(xs)
    if not all((isinstance(xs, torch.Tensor) for xs in flat_xs)):
        raise RuntimeError(f'Leaves of xs must be Tensor {flat_xs}')
    if not all((xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs)):
        raise RuntimeError(f'Leaves of xs must have same leading dimension size {[xs.shape for xs in flat_xs]}')
    a = zip(*flat_xs)
    pytrees = []
    for tuple in a:
        pytrees.append(pytree.tree_unflatten(tuple, inspec))
    return pytrees