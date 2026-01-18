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
def _get_view_type(tgt) -> _ViewType:
    if tgt is not None and isinstance(tgt, torch._ops.OpOverload):
        schema = tgt._schema
        if len(schema.arguments) > 0:
            first_arg = schema.arguments[0]
            if first_arg.alias_info is not None and (not first_arg.alias_info.is_write):
                if '*' in first_arg.alias_info.after_set:
                    return _ViewType.MultiOutputView
                else:
                    return _ViewType.SingleOutputView
    return _ViewType.NonView