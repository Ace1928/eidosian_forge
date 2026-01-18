from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
class BadAttr2(float):

    def __getattribute__(self, key):
        assert False