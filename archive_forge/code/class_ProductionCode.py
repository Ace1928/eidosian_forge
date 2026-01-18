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
class ProductionCode:

    def __init__(self):
        self.session = None

    async def main(self):
        async with self.session.post('https://python.org') as response:
            val = await response.json()
            return val