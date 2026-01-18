from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
class PropertyMock(Mock):
    """
    A mock intended to be used as a property, or other descriptor, on a class.
    `PropertyMock` provides `__get__` and `__set__` methods so you can specify
    a return value when it is fetched.

    Fetching a `PropertyMock` instance from an object calls the mock, with
    no args. Setting it calls the mock with the value being set.
    """

    def _get_child_mock(self, **kwargs):
        return MagicMock(**kwargs)

    def __get__(self, obj, obj_type):
        return self()

    def __set__(self, obj, val):
        self(val)