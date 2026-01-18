from __future__ import annotations
import warnings
from .logger import adapter as logger
from .logger import trace as _trace
import os
import sys
import builtins as __builtin__
from pickle import _Pickler as StockPickler, Unpickler as StockUnpickler
from pickle import GLOBAL, POP
from _thread import LockType
from _thread import RLock as RLockType
from types import CodeType, FunctionType, MethodType, GeneratorType, \
from types import MappingProxyType as DictProxyType, new_class
from pickle import DEFAULT_PROTOCOL, HIGHEST_PROTOCOL, PickleError, PicklingError, UnpicklingError
import __main__ as _main_module
import marshal
import gc
import abc
import dataclasses
from weakref import ReferenceType, ProxyType, CallableProxyType
from collections import OrderedDict
from enum import Enum, EnumMeta
from functools import partial
from operator import itemgetter, attrgetter
import importlib.machinery
from types import GetSetDescriptorType, ClassMethodDescriptorType, \
from io import BytesIO as StringIO
from socket import socket as SocketType
from multiprocessing.reduction import _reduce_socket as reduce_socket
import inspect
import typing
from . import _shims
from ._shims import Reduce, Getattr
def _create_code(*args):
    if not isinstance(args[0], int):
        LNOTAB, *args = args
    else:
        LNOTAB = b''
    with match(args) as m:
        if m.case(('argcount', 'posonlyargcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags', 'code', 'consts', 'names', 'varnames', 'filename', 'name', 'qualname', 'firstlineno', 'linetable', 'exceptiontable', 'freevars', 'cellvars')):
            if CODE_VERSION == (3, 11):
                return CodeType(*args[:6], args[6].encode() if hasattr(args[6], 'encode') else args[6], *args[7:14], args[14].encode() if hasattr(args[14], 'encode') else args[14], args[15].encode() if hasattr(args[15], 'encode') else args[15], args[16], args[17])
            fields = m.fields
        elif m.case(('argcount', 'posonlyargcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags', 'code', 'consts', 'names', 'varnames', 'filename', 'name', 'firstlineno', 'LNOTAB_OR_LINETABLE', 'freevars', 'cellvars')):
            if CODE_VERSION == (3, 10) or CODE_VERSION == (3, 8):
                return CodeType(*args[:6], args[6].encode() if hasattr(args[6], 'encode') else args[6], *args[7:13], args[13].encode() if hasattr(args[13], 'encode') else args[13], args[14], args[15])
            fields = m.fields
            if CODE_VERSION >= (3, 10):
                fields['linetable'] = m.LNOTAB_OR_LINETABLE
            else:
                fields['lnotab'] = LNOTAB if LNOTAB else m.LNOTAB_OR_LINETABLE
        elif m.case(('argcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags', 'code', 'consts', 'names', 'varnames', 'filename', 'name', 'firstlineno', 'lnotab', 'freevars', 'cellvars')):
            if CODE_VERSION == (3, 7):
                return CodeType(*args[:5], args[5].encode() if hasattr(args[5], 'encode') else args[5], *args[6:12], args[12].encode() if hasattr(args[12], 'encode') else args[12], args[13], args[14])
            fields = m.fields
        elif m.case(('argcount', 'posonlyargcount', 'kwonlyargcount', 'nlocals', 'stacksize', 'flags', 'code', 'consts', 'names', 'varnames', 'filename', 'name', 'qualname', 'firstlineno', 'linetable', 'endlinetable', 'columntable', 'exceptiontable', 'freevars', 'cellvars')):
            if CODE_VERSION == (3, 11, 'a'):
                return CodeType(*args[:6], args[6].encode() if hasattr(args[6], 'encode') else args[6], *args[7:14], *(a.encode() if hasattr(a, 'encode') else a for a in args[14:18]), args[18], args[19])
            fields = m.fields
        else:
            raise UnpicklingError('pattern match for code object failed')
    fields.setdefault('posonlyargcount', 0)
    fields.setdefault('lnotab', LNOTAB)
    fields.setdefault('linetable', b'')
    fields.setdefault('qualname', fields['name'])
    fields.setdefault('exceptiontable', b'')
    fields.setdefault('endlinetable', None)
    fields.setdefault('columntable', None)
    args = (fields[k].encode() if k in ENCODE_PARAMS and hasattr(fields[k], 'encode') else fields[k] for k in CODE_PARAMS)
    return CodeType(*args)