import asyncio
import contextlib
import io
import inspect
import pprint
import sys
import builtins
import pkgutil
from asyncio import iscoroutinefunction
from types import CodeType, ModuleType, MethodType
from unittest.util import safe_repr
from functools import wraps, partial
from threading import RLock
class AsyncMockMixin(Base):
    await_count = _delegating_property('await_count')
    await_args = _delegating_property('await_args')
    await_args_list = _delegating_property('await_args_list')

    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__['_is_coroutine'] = asyncio.coroutines._is_coroutine
        self.__dict__['_mock_await_count'] = 0
        self.__dict__['_mock_await_args'] = None
        self.__dict__['_mock_await_args_list'] = _CallList()
        code_mock = NonCallableMock(spec_set=CodeType)
        code_mock.co_flags = inspect.CO_COROUTINE + inspect.CO_VARARGS + inspect.CO_VARKEYWORDS
        code_mock.co_argcount = 0
        code_mock.co_varnames = ('args', 'kwargs')
        code_mock.co_posonlyargcount = 0
        code_mock.co_kwonlyargcount = 0
        self.__dict__['__code__'] = code_mock
        self.__dict__['__name__'] = 'AsyncMock'
        self.__dict__['__defaults__'] = tuple()
        self.__dict__['__kwdefaults__'] = {}
        self.__dict__['__annotations__'] = None

    async def _execute_mock_call(self, /, *args, **kwargs):
        _call = _Call((args, kwargs), two=True)
        self.await_count += 1
        self.await_args = _call
        self.await_args_list.append(_call)
        effect = self.side_effect
        if effect is not None:
            if _is_exception(effect):
                raise effect
            elif not _callable(effect):
                try:
                    result = next(effect)
                except StopIteration:
                    raise StopAsyncIteration
                if _is_exception(result):
                    raise result
            elif iscoroutinefunction(effect):
                result = await effect(*args, **kwargs)
            else:
                result = effect(*args, **kwargs)
            if result is not DEFAULT:
                return result
        if self._mock_return_value is not DEFAULT:
            return self.return_value
        if self._mock_wraps is not None:
            if iscoroutinefunction(self._mock_wraps):
                return await self._mock_wraps(*args, **kwargs)
            return self._mock_wraps(*args, **kwargs)
        return self.return_value

    def assert_awaited(self):
        """
        Assert that the mock was awaited at least once.
        """
        if self.await_count == 0:
            msg = f'Expected {self._mock_name or 'mock'} to have been awaited.'
            raise AssertionError(msg)

    def assert_awaited_once(self):
        """
        Assert that the mock was awaited exactly once.
        """
        if not self.await_count == 1:
            msg = f'Expected {self._mock_name or 'mock'} to have been awaited once. Awaited {self.await_count} times.'
            raise AssertionError(msg)

    def assert_awaited_with(self, /, *args, **kwargs):
        """
        Assert that the last await was with the specified arguments.
        """
        if self.await_args is None:
            expected = self._format_mock_call_signature(args, kwargs)
            raise AssertionError(f'Expected await: {expected}\nNot awaited')

        def _error_message():
            msg = self._format_mock_failure_message(args, kwargs, action='await')
            return msg
        expected = self._call_matcher(_Call((args, kwargs), two=True))
        actual = self._call_matcher(self.await_args)
        if actual != expected:
            cause = expected if isinstance(expected, Exception) else None
            raise AssertionError(_error_message()) from cause

    def assert_awaited_once_with(self, /, *args, **kwargs):
        """
        Assert that the mock was awaited exactly once and with the specified
        arguments.
        """
        if not self.await_count == 1:
            msg = f'Expected {self._mock_name or 'mock'} to have been awaited once. Awaited {self.await_count} times.'
            raise AssertionError(msg)
        return self.assert_awaited_with(*args, **kwargs)

    def assert_any_await(self, /, *args, **kwargs):
        """
        Assert the mock has ever been awaited with the specified arguments.
        """
        expected = self._call_matcher(_Call((args, kwargs), two=True))
        cause = expected if isinstance(expected, Exception) else None
        actual = [self._call_matcher(c) for c in self.await_args_list]
        if cause or expected not in _AnyComparer(actual):
            expected_string = self._format_mock_call_signature(args, kwargs)
            raise AssertionError('%s await not found' % expected_string) from cause

    def assert_has_awaits(self, calls, any_order=False):
        """
        Assert the mock has been awaited with the specified calls.
        The :attr:`await_args_list` list is checked for the awaits.

        If `any_order` is False (the default) then the awaits must be
        sequential. There can be extra calls before or after the
        specified awaits.

        If `any_order` is True then the awaits can be in any order, but
        they must all appear in :attr:`await_args_list`.
        """
        expected = [self._call_matcher(c) for c in calls]
        cause = next((e for e in expected if isinstance(e, Exception)), None)
        all_awaits = _CallList((self._call_matcher(c) for c in self.await_args_list))
        if not any_order:
            if expected not in all_awaits:
                if cause is None:
                    problem = 'Awaits not found.'
                else:
                    problem = 'Error processing expected awaits.\nErrors: {}'.format([e if isinstance(e, Exception) else None for e in expected])
                raise AssertionError(f'{problem}\nExpected: {_CallList(calls)}\nActual: {self.await_args_list}') from cause
            return
        all_awaits = list(all_awaits)
        not_found = []
        for kall in expected:
            try:
                all_awaits.remove(kall)
            except ValueError:
                not_found.append(kall)
        if not_found:
            raise AssertionError('%r not all found in await list' % (tuple(not_found),)) from cause

    def assert_not_awaited(self):
        """
        Assert that the mock was never awaited.
        """
        if self.await_count != 0:
            msg = f'Expected {self._mock_name or 'mock'} to not have been awaited. Awaited {self.await_count} times.'
            raise AssertionError(msg)

    def reset_mock(self, /, *args, **kwargs):
        """
        See :func:`.Mock.reset_mock()`
        """
        super().reset_mock(*args, **kwargs)
        self.await_count = 0
        self.await_args = None
        self.await_args_list = _CallList()