import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def _get_proxy(obj, get_only=True):

    class Proxy(object):

        def __getattr__(self, name):
            return getattr(obj, name)
    if not get_only:

        def __setattr__(self, name, value):
            setattr(obj, name, value)

        def __delattr__(self, name):
            delattr(obj, name)
        Proxy.__setattr__ = __setattr__
        Proxy.__delattr__ = __delattr__
    return Proxy()