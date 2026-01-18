from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _dot_lookup(thing, comp, import_path):
    try:
        return getattr(thing, comp)
    except AttributeError:
        __import__(import_path)
        return getattr(thing, comp)