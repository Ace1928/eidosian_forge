from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
class EnumMeta:
    """ This is just an EnumMeta stub for Python 2 and Python 3.3 and before without Enum support. """