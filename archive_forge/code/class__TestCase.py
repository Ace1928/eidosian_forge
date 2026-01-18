import contextlib
import copy
import pickle
import unittest
from types import FunctionType, ModuleType
from typing import Any, Dict, Set
from unittest import mock
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