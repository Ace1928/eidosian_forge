from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _unpatch_dict(self):
    in_dict = self.in_dict
    original = self._original
    _clear_dict(in_dict)
    try:
        in_dict.update(original)
    except AttributeError:
        for key in original:
            in_dict[key] = original[key]