import collections
import doctest
import types
from typing import Any, Iterator, Mapping
import unittest
from absl.testing import parameterized
import attr
import numpy as np
import tree
import wrapt
class BadAttr(object):
    """Class that has a non-iterable __attrs_attrs__."""
    __attrs_attrs__ = None