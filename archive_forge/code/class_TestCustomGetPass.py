from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
class TestCustomGetPass(GetFunctionTestMixin):
    get = staticmethod(get)