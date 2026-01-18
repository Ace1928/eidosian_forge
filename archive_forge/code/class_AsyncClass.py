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
class AsyncClass:

    def __init__(self):
        pass

    async def async_method(self):
        pass

    def normal_method(self):
        pass

    @classmethod
    async def async_class_method(cls):
        pass

    @staticmethod
    async def async_static_method():
        pass