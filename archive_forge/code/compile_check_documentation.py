import torch
from .make_fx import make_fx_check
from .aot_autograd import aot_autograd_check
from .fake_tensor import fake_check
from torch._subclasses.schema_check_mode import SchemaCheckMode
Check if torch.compile supports a custom operator.

    Args:
        func (function): an operator that takes at least one Tensor as input
            and returns a Tensor or a Tuple of Tensors.
        args (Tuple): args to the operator
        kwargs (dict, optional): kwargs to the operator
        dynamic_only (bool, optional): If the operator only works with dynamic
            shapes. This can happen if it returns Tensors whose shape are
            dependent on the data on the input Tensors. If True, we skip
            tests related to torch.compile with static shapes.
        supports_autograd (bool, optional): If the operator does not support
            autograd. If False, we will skip autograd-related tests.
        fullgraph (bool, optional): If we expect torch.compile to not graph
            break on seeing the operator. Graph breaks are bad for performance.

    