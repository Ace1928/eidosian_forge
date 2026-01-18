from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _is_magic(name):
    return '__%s__' % name[2:-2] == name