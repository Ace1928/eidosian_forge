import abc
import typing as t
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from .shape_prop import TensorMetadata
from .tools_common import get_node_target, CALLABLE_NODE_OPS
def _get_arg_dtype(arg: torch.fx.Node) -> t.Any:
    assert isinstance(arg, torch.fx.Node)
    tensor_meta = arg.meta.get('tensor_meta')
    dtype = tensor_meta.dtype if isinstance(tensor_meta, TensorMetadata) else arg.meta['type']
    return dtype