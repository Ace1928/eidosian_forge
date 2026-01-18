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
def _make_dict_values(obj, is_ordered=False):
    if is_ordered:
        return OrderedDict(((i, _) for i, _ in enumerate(obj))).values()
    else:
        return {i: _ for i, _ in enumerate(obj)}.values()