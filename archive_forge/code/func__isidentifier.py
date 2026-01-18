from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _isidentifier(string):
    if string in keyword.kwlist:
        return False
    return regex.match(string)