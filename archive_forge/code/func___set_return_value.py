from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def __set_return_value(self, value):
    if self._mock_delegate is not None:
        self._mock_delegate.return_value = value
    else:
        self._mock_return_value = value
        _check_and_set_parent(self, value, None, '()')