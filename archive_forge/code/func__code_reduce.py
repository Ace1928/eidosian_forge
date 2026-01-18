import _collections_abc
import abc
import copyreg
import io
import itertools
import logging
import sys
import struct
import types
import weakref
import typing
from enum import Enum
from collections import ChainMap, OrderedDict
from .compat import pickle, Pickler
from .cloudpickle import (
def _code_reduce(obj):
    """codeobject reducer"""
    if hasattr(obj, 'co_exceptiontable'):
        args = (obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_qualname, obj.co_firstlineno, obj.co_linetable, obj.co_exceptiontable, obj.co_freevars, obj.co_cellvars)
    elif hasattr(obj, 'co_linetable'):
        args = (obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_linetable, obj.co_freevars, obj.co_cellvars)
    elif hasattr(obj, 'co_nmeta'):
        args = (obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_framesize, obj.co_ndefaultargs, obj.co_nmeta, obj.co_flags, obj.co_code, obj.co_consts, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_exc_handlers, obj.co_jump_table, obj.co_freevars, obj.co_cellvars, obj.co_free2reg, obj.co_cell2reg)
    elif hasattr(obj, 'co_posonlyargcount'):
        args = (obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars)
    else:
        args = (obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars)
    return (types.CodeType, args)