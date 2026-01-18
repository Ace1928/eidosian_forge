from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _check_and_set_parent(parent, value, name, new_name):
    if not _is_instance_mock(value):
        return False
    if (value._mock_name or value._mock_new_name) or value._mock_parent is not None or value._mock_new_parent is not None:
        return False
    _parent = parent
    while _parent is not None:
        if _parent is value:
            return False
        _parent = _parent._mock_new_parent
    if new_name:
        value._mock_new_parent = parent
        value._mock_new_name = new_name
    if name:
        value._mock_parent = parent
        value._mock_name = name
    return True