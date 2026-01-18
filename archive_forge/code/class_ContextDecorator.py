import contextlib
import copy
import pickle
import unittest
from types import FunctionType, ModuleType
from typing import Any, Dict, Set
from unittest import mock
class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """

    def __call__(self, func):
        if isinstance(func, type) and issubclass(func, unittest.TestCase):

            class _TestCase(func):

                @classmethod
                def setUpClass(cls):
                    self.__enter__()
                    try:
                        super().setUpClass()
                    except Exception:
                        self.__exit__(None, None, None)
                        raise

                @classmethod
                def tearDownClass(cls):
                    try:
                        super().tearDownClass()
                    finally:
                        self.__exit__(None, None, None)
            _TestCase.__name__ = func.__name__
            _TestCase.__qualname__ = func.__qualname__
            _TestCase.__module__ = func.__module__
            return _TestCase
        return super().__call__(func)