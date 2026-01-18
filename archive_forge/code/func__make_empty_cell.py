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
def _make_empty_cell():
    if False:
        cell = None
        raise AssertionError('this route should not be executed')
    return (lambda: cell).__closure__[0]