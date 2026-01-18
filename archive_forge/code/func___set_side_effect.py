from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def __set_side_effect(self, value):
    value = _try_iter(value)
    delegated = self._mock_delegate
    if delegated is None:
        self._mock_side_effect = value
    else:
        delegated.side_effect = value