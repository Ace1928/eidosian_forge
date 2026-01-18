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
class AsyncContextManagerTest(unittest.TestCase):

    class WithAsyncContextManager:

        async def __aenter__(self, *args, **kwargs):
            pass

        async def __aexit__(self, *args, **kwargs):
            pass

    class WithSyncContextManager:

        def __enter__(self, *args, **kwargs):
            pass

        def __exit__(self, *args, **kwargs):
            pass

    class ProductionCode:

        def __init__(self):
            self.session = None

        async def main(self):
            async with self.session.post('https://python.org') as response:
                val = await response.json()
                return val

    def test_set_return_value_of_aenter(self):

        def inner_test(mock_type):
            pc = self.ProductionCode()
            pc.session = MagicMock(name='sessionmock')
            cm = mock_type(name='magic_cm')
            response = AsyncMock(name='response')
            response.json = AsyncMock(return_value={'json': 123})
            cm.__aenter__.return_value = response
            pc.session.post.return_value = cm
            result = run(pc.main())
            self.assertEqual(result, {'json': 123})
        for mock_type in [AsyncMock, MagicMock]:
            with self.subTest(f'test set return value of aenter with {mock_type}'):
                inner_test(mock_type)

    def test_mock_supports_async_context_manager(self):

        def inner_test(mock_type):
            called = False
            cm = self.WithAsyncContextManager()
            cm_mock = mock_type(cm)

            async def use_context_manager():
                nonlocal called
                async with cm_mock as result:
                    called = True
                return result
            cm_result = run(use_context_manager())
            self.assertTrue(called)
            self.assertTrue(cm_mock.__aenter__.called)
            self.assertTrue(cm_mock.__aexit__.called)
            cm_mock.__aenter__.assert_awaited()
            cm_mock.__aexit__.assert_awaited()
            self.assertIsNot(cm_mock, cm_result)
        for mock_type in [AsyncMock, MagicMock]:
            with self.subTest(f'test context manager magics with {mock_type}'):
                inner_test(mock_type)

    def test_mock_customize_async_context_manager(self):
        instance = self.WithAsyncContextManager()
        mock_instance = MagicMock(instance)
        expected_result = object()
        mock_instance.__aenter__.return_value = expected_result

        async def use_context_manager():
            async with mock_instance as result:
                return result
        self.assertIs(run(use_context_manager()), expected_result)

    def test_mock_customize_async_context_manager_with_coroutine(self):
        enter_called = False
        exit_called = False

        async def enter_coroutine(*args):
            nonlocal enter_called
            enter_called = True

        async def exit_coroutine(*args):
            nonlocal exit_called
            exit_called = True
        instance = self.WithAsyncContextManager()
        mock_instance = MagicMock(instance)
        mock_instance.__aenter__ = enter_coroutine
        mock_instance.__aexit__ = exit_coroutine

        async def use_context_manager():
            async with mock_instance:
                pass
        run(use_context_manager())
        self.assertTrue(enter_called)
        self.assertTrue(exit_called)

    def test_context_manager_raise_exception_by_default(self):

        async def raise_in(context_manager):
            async with context_manager:
                raise TypeError()
        instance = self.WithAsyncContextManager()
        mock_instance = MagicMock(instance)
        with self.assertRaises(TypeError):
            run(raise_in(mock_instance))