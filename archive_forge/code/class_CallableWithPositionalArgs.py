from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class CallableWithPositionalArgs(object):
    """Test class for supporting callable."""
    TEST = 1

    def __call__(self, x, y):
        return x + y

    def fn(self, x):
        return x + 1