from functools import lru_cache
from itertools import chain
from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import TensorMeta
from torch.distributed.device_mesh import DeviceMesh
def _propagate_tensor_meta(self, op_schema: OpSchema) -> Union[None, TensorMeta, List[TensorMeta], Tuple[TensorMeta, ...]]:
    """
        Propagate the tensor metadata, it could either return a TensorMeta
        or a list/tuple of TensorMetas
        """
    if op_schema.op == aten.equal.default:
        return None
    with FakeTensorMode():
        fake_args = op_schema.gen_fake_args()
        fake_kwargs = op_schema.gen_fake_kwargs()
        fake_out = op_schema.op(*fake_args, **fake_kwargs)
    if isinstance(fake_out, torch.Tensor):
        return TensorMeta(shape=fake_out.shape, stride=fake_out.stride(), dtype=fake_out.dtype)
    elif isinstance(fake_out, (tuple, list)):
        tensor_meta_list = []
        for fake_out_item in fake_out:
            if isinstance(fake_out_item, torch.Tensor):
                tensor_meta_list.append(TensorMeta(shape=fake_out_item.shape, stride=fake_out_item.stride(), dtype=fake_out_item.dtype))
        return tuple(tensor_meta_list) if isinstance(fake_out, tuple) else tensor_meta_list
    else:
        return None