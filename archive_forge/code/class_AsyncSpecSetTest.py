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
class AsyncSpecSetTest(unittest.TestCase):

    def test_is_AsyncMock_patch(self):

        @patch.object(AsyncClass, 'async_method', spec_set=True)
        def test_async(async_method):
            self.assertIsInstance(async_method, AsyncMock)
        test_async()

    def test_is_async_AsyncMock(self):
        mock = AsyncMock(spec_set=AsyncClass.async_method)
        self.assertTrue(iscoroutinefunction(mock))
        self.assertIsInstance(mock, AsyncMock)

    def test_is_child_AsyncMock(self):
        mock = MagicMock(spec_set=AsyncClass)
        self.assertTrue(iscoroutinefunction(mock.async_method))
        self.assertFalse(iscoroutinefunction(mock.normal_method))
        self.assertIsInstance(mock.async_method, AsyncMock)
        self.assertIsInstance(mock.normal_method, MagicMock)
        self.assertIsInstance(mock, MagicMock)

    def test_magicmock_lambda_spec(self):
        mock_obj = MagicMock()
        mock_obj.mock_func = MagicMock(spec=lambda x: x)
        with patch.object(mock_obj, 'mock_func') as cm:
            self.assertIsInstance(cm, MagicMock)