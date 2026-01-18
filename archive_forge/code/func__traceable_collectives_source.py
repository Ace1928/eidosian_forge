import functools
import inspect
import itertools
import types
from typing import Dict, List
import torch
from .. import variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented, Unsupported
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import make_cell
from .base import typestr, VariableTracker
def _traceable_collectives_source(tx, fn):
    assert torch.distributed.is_available(), 'Illegal invocation.'
    from torch.distributed._functional_collectives import all_gather_tensor_inplace, reduce_scatter_tensor_inplace
    valid_values = {all_gather_tensor_inplace, reduce_scatter_tensor_inplace}
    assert fn in valid_values
    inner_name = fn.__name__
    path_source = tx.import_source('torch.distributed._functional_collectives')
    return AttrSource(path_source, inner_name)