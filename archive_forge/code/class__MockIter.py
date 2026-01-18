from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
class _MockIter(object):

    def __init__(self, obj):
        self.obj = iter(obj)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.obj)