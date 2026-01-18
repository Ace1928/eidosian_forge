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
def _odict_keys_reduce(obj):
    return (_make_dict_keys, (list(obj), True))