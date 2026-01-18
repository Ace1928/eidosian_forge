import math
import ctypes
import copy
from .random import uniform
from .symbol import Symbol
from . import symbol
from ..base import _LIB, check_call
from ..base import SymbolHandle, _as_list
from ..attribute import AttrScope
def _get_unique_subgraph_name(subgraph_name):
    attrs = AttrScope._current.value._attr
    if attrs.get('__subgraph_name__', '') != '':
        subgraph_name = ''.join([attrs['__subgraph_name__'], '$', subgraph_name])
    AttrScope._subgraph_names[subgraph_name] += 1
    subgraph_name = subgraph_name + str(AttrScope._subgraph_names[subgraph_name] - 1)
    return subgraph_name