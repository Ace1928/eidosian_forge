import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
class Countdown:

    def __init__(self, count):
        self.count = count
        self.closed = False

    async def __aiter__(self):
        return self

    async def __anext__(self):
        self.count -= 1
        if self.count < 0:
            raise StopAsyncIteration('boom')
        return self.count