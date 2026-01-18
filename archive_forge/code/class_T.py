from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
class T:

    def __getitem__(self, k):
        return 'a'

    def __getattr__(self, k):
        return 'a'