from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _read_side_effect(*args, **kwargs):
    if handle.read.return_value is not None:
        return handle.read.return_value
    return type(read_data)().join(_state[0])