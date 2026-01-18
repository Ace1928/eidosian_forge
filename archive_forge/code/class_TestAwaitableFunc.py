import abc
import concurrent.futures
import contextlib
import inspect
import sys
import time
import traceback
from typing import List, Tuple
import pytest
import duet
import duet.impl as impl
class TestAwaitableFunc:

    def test_wrap_async_func(self):

        async def async_func(a, b):
            await duet.completed_future(None)
            return a + b
        assert duet.awaitable_func(async_func) is async_func
        assert duet.run(async_func, 1, 2) == 3

    def test_wrap_sync_func(self):

        def sync_func(a, b):
            return a + b
        wrapped = duet.awaitable_func(sync_func)
        assert inspect.iscoroutinefunction(wrapped)
        assert duet.awaitable_func(wrapped) is wrapped
        assert duet.run(wrapped, 1, 2) == 3