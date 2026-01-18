from __future__ import absolute_import, print_function, unicode_literals
import typing
import abc
import hashlib
import itertools
import os
import six
import threading
import time
import warnings
from contextlib import closing
from functools import partial, wraps
from . import copy, errors, fsencode, iotools, tools, walk, wildcard
from .copy import copy_modified_time
from .glob import BoundGlobber
from .mode import validate_open_mode
from .path import abspath, join, normpath
from .time import datetime_to_epoch
from .walk import Walker
def _new_name(method, old_name):
    """Return a method with a deprecation warning."""

    @wraps(method)
    def _method(*args, **kwargs):
        warnings.warn("method '{}' has been deprecated, please rename to '{}'".format(old_name, method.__name__), DeprecationWarning)
        return method(*args, **kwargs)
    deprecated_msg = '\n        Note:\n            .. deprecated:: 2.2.0\n                Please use `~{}`\n'.format(method.__name__)
    if getattr(_method, '__doc__', None) is not None:
        _method.__doc__ += deprecated_msg
    return _method