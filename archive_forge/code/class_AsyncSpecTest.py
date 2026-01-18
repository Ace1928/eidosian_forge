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
class AsyncSpecTest(unittest.TestCase):

    def test_spec_normal_methods_on_class(self):

        def inner_test(mock_type):
            mock = mock_type(AsyncClass)
            self.assertIsInstance(mock.async_method, AsyncMock)
            self.assertIsInstance(mock.normal_method, MagicMock)
        for mock_type in [AsyncMock, MagicMock]:
            with self.subTest(f'test method types with {mock_type}'):
                inner_test(mock_type)

    def test_spec_normal_methods_on_class_with_mock(self):
        mock = Mock(AsyncClass)
        self.assertIsInstance(mock.async_method, AsyncMock)
        self.assertIsInstance(mock.normal_method, Mock)

    def test_spec_normal_methods_on_class_with_mock_seal(self):
        mock = Mock(AsyncClass)
        seal(mock)
        with self.assertRaises(AttributeError):
            mock.normal_method
        with self.assertRaises(AttributeError):
            mock.async_method

    def test_spec_mock_type_kw(self):

        def inner_test(mock_type):
            async_mock = mock_type(spec=async_func)
            self.assertIsInstance(async_mock, mock_type)
            with assertNeverAwaited(self):
                self.assertTrue(inspect.isawaitable(async_mock()))
            sync_mock = mock_type(spec=normal_func)
            self.assertIsInstance(sync_mock, mock_type)
        for mock_type in [AsyncMock, MagicMock, Mock]:
            with self.subTest(f'test spec kwarg with {mock_type}'):
                inner_test(mock_type)

    def test_spec_mock_type_positional(self):

        def inner_test(mock_type):
            async_mock = mock_type(async_func)
            self.assertIsInstance(async_mock, mock_type)
            with assertNeverAwaited(self):
                self.assertTrue(inspect.isawaitable(async_mock()))
            sync_mock = mock_type(normal_func)
            self.assertIsInstance(sync_mock, mock_type)
        for mock_type in [AsyncMock, MagicMock, Mock]:
            with self.subTest(f'test spec positional with {mock_type}'):
                inner_test(mock_type)

    def test_spec_as_normal_kw_AsyncMock(self):
        mock = AsyncMock(spec=normal_func)
        self.assertIsInstance(mock, AsyncMock)
        m = mock()
        self.assertTrue(inspect.isawaitable(m))
        run(m)

    def test_spec_as_normal_positional_AsyncMock(self):
        mock = AsyncMock(normal_func)
        self.assertIsInstance(mock, AsyncMock)
        m = mock()
        self.assertTrue(inspect.isawaitable(m))
        run(m)

    def test_spec_async_mock(self):

        @patch.object(AsyncClass, 'async_method', spec=True)
        def test_async(mock_method):
            self.assertIsInstance(mock_method, AsyncMock)
        test_async()

    def test_spec_parent_not_async_attribute_is(self):

        @patch(async_foo_name, spec=True)
        def test_async(mock_method):
            self.assertIsInstance(mock_method, MagicMock)
            self.assertIsInstance(mock_method.async_method, AsyncMock)
        test_async()

    def test_target_async_spec_not(self):

        @patch.object(AsyncClass, 'async_method', spec=NormalClass.a)
        def test_async_attribute(mock_method):
            self.assertIsInstance(mock_method, MagicMock)
            self.assertFalse(inspect.iscoroutine(mock_method))
            self.assertFalse(inspect.isawaitable(mock_method))
        test_async_attribute()

    def test_target_not_async_spec_is(self):

        @patch.object(NormalClass, 'a', spec=async_func)
        def test_attribute_not_async_spec_is(mock_async_func):
            self.assertIsInstance(mock_async_func, AsyncMock)
        test_attribute_not_async_spec_is()

    def test_spec_async_attributes(self):

        @patch(normal_foo_name, spec=AsyncClass)
        def test_async_attributes_coroutines(MockNormalClass):
            self.assertIsInstance(MockNormalClass.async_method, AsyncMock)
            self.assertIsInstance(MockNormalClass, MagicMock)
        test_async_attributes_coroutines()