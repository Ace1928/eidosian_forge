import ctypes
import logging
import os
import shutil
import warnings
import numpy as np
from ..base import _LIB, check_call, py_str
from ..base import c_array, c_str, mx_uint, c_str_array
from ..base import NDArrayHandle, SymbolHandle
from ..symbol import Symbol
from ..symbol import load as sym_load
from .. import ndarray
from ..ndarray import load as nd_load
from ..ndarray import save as nd_save
from ..ndarray import NDArray
from ..io import DataIter, DataDesc, DataBatch
from ..context import cpu, Context
from ..module import Module
def _load_sym(sym, logger=None):
    """Given a str as a path the symbol .json file or a symbol, returns a Symbol object."""
    if isinstance(sym, str):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        symbol_file_path = os.path.join(cur_path, sym)
        if logger:
            logger.info('Loading symbol from file %s' % symbol_file_path)
        return sym_load(symbol_file_path)
    elif isinstance(sym, Symbol):
        return sym
    else:
        raise ValueError('_load_sym only accepts Symbol or path to the symbol file, while received type %s' % str(type(sym)))