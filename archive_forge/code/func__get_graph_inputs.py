import math
import ctypes
import copy
from .random import uniform
from .symbol import Symbol
from . import symbol
from ..base import _LIB, check_call
from ..base import SymbolHandle, _as_list
from ..attribute import AttrScope
def _get_graph_inputs(subg):
    num_handles = ctypes.c_int(0)
    handles = ctypes.POINTER(SymbolHandle)()
    check_call(_LIB.MXSymbolGetInputSymbols(subg.handle, ctypes.byref(handles), ctypes.byref(num_handles)))
    syms = []
    for i in range(num_handles.value):
        s = Symbol(ctypes.cast(handles[i], SymbolHandle))
        syms.append(s)
    return syms