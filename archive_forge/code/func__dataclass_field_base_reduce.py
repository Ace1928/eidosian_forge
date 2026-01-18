import _collections_abc
from collections import ChainMap, OrderedDict
import abc
import builtins
import copyreg
import dataclasses
import dis
from enum import Enum
import io
import itertools
import logging
import opcode
import pickle
from pickle import _getattribute
import platform
import struct
import sys
import threading
import types
import typing
import uuid
import warnings
import weakref
from types import CellType  # noqa: F401
def _dataclass_field_base_reduce(obj):
    return (_get_dataclass_field_type_sentinel, (obj.name,))