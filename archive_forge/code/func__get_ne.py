from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _get_ne(self):

    def __ne__(other):
        if self.__ne__._mock_return_value is not DEFAULT:
            return DEFAULT
        return self is not other
    return __ne__