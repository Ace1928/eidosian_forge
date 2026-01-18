from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
class Custom:

    def __init__(self):
        self.test = 1

    def __getattr__(self, key):
        return key