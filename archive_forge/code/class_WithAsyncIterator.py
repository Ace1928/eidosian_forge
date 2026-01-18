import asyncio
import gc
import inspect
import re
import unittest
from contextlib import contextmanager
from test import support
from asyncio import run, iscoroutinefunction
from unittest import IsolatedAsyncioTestCase
from unittest.mock import (ANY, call, AsyncMock, patch, MagicMock, Mock,
class WithAsyncIterator(object):

    def __init__(self):
        self.items = ['foo', 'NormalFoo', 'baz']

    def __aiter__(self):
        pass

    async def __anext__(self):
        pass