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
def _make_dict_keys(obj, is_ordered=False):
    if is_ordered:
        return OrderedDict.fromkeys(obj).keys()
    else:
        return dict.fromkeys(obj).keys()