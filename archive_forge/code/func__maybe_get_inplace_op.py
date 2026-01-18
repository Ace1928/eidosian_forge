import torch
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map_only
from torch.utils import _pytree as pytree
from torch.multiprocessing.reductions import StorageWeakRef
import _operator
from enum import Enum
import itertools
from typing import Set, Dict
from collections import defaultdict
def _maybe_get_inplace_op(op):
    if not isinstance(op, torch._ops.OpOverload):
        return None
    if _is_view_op(op):
        return None
    op_namespace = op.__module__.split('.')[-1]
    op_base_name = op.overloadpacket.__name__
    maybe_namespace_module = getattr(torch.ops, op_namespace)
    maybe_inplace_op = None if maybe_namespace_module is None else getattr(maybe_namespace_module, f'{op_base_name}_', None)
    if maybe_inplace_op is None:
        return None
    inplace_overloads = [getattr(maybe_inplace_op, overload_name) for overload_name in maybe_inplace_op.overloads()]
    inplace_overloads_with_matching_schemas = [f for f in inplace_overloads if _schemas_match(op._schema, f._schema)]
    if len(inplace_overloads_with_matching_schemas) == 0:
        return None
    assert len(inplace_overloads_with_matching_schemas) == 1
    inplace_op = inplace_overloads_with_matching_schemas[0]
    return inplace_op