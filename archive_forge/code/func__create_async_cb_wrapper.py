import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
@staticmethod
def _create_async_cb_wrapper(callback, /, *args, **kwds):

    async def _exit_wrapper(exc_type, exc, tb):
        await callback(*args, **kwds)
    return _exit_wrapper