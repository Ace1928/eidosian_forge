from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
class ObjectProxy(object):
    """Stub-class for `wrapt.ObjectProxy``."""