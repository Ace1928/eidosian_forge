import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings
from .compat import pickle
from collections import OrderedDict
from typing import ClassVar, Generic, Union, Tuple, Callable
from pickle import _getattribute
def _fill_function(*args):
    """Fills in the rest of function data into the skeleton function object

    The skeleton itself is create by _make_skel_func().
    """
    if len(args) == 2:
        func = args[0]
        state = args[1]
    elif len(args) == 5:
        func = args[0]
        keys = ['globals', 'defaults', 'dict', 'closure_values']
        state = dict(zip(keys, args[1:]))
    elif len(args) == 6:
        func = args[0]
        keys = ['globals', 'defaults', 'dict', 'module', 'closure_values']
        state = dict(zip(keys, args[1:]))
    else:
        raise ValueError(f'Unexpected _fill_value arguments: {args!r}')
    func.__globals__.update(state['globals'])
    func.__defaults__ = state['defaults']
    func.__dict__ = state['dict']
    if 'annotations' in state:
        func.__annotations__ = state['annotations']
    if 'doc' in state:
        func.__doc__ = state['doc']
    if 'name' in state:
        func.__name__ = state['name']
    if 'module' in state:
        func.__module__ = state['module']
    if 'qualname' in state:
        func.__qualname__ = state['qualname']
    if 'kwdefaults' in state:
        func.__kwdefaults__ = state['kwdefaults']
    if '_cloudpickle_submodules' in state:
        state.pop('_cloudpickle_submodules')
    cells = func.__closure__
    if cells is not None:
        for cell, value in zip(cells, state['closure_values']):
            if value is not _empty_cell_value:
                cell_set(cell, value)
    return func