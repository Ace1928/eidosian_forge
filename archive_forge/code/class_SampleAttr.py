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
@attr.s
class SampleAttr(object):
    field3 = attr.ib()
    field1 = attr.ib()
    field2 = attr.ib()