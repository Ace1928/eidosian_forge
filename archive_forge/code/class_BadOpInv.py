from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
class BadOpInv(int):

    def __inv__(self, other):
        assert False