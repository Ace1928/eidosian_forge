import sys
from contextlib import contextmanager
from typing import (
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
class TestProtocolImplementer(TestProtocol):

    def test_method(self) -> bool:
        return True