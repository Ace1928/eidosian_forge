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
def _decompose_typevar(obj):
    return (obj.__name__, obj.__bound__, obj.__constraints__, obj.__covariant__, obj.__contravariant__, _get_or_create_tracker_id(obj))