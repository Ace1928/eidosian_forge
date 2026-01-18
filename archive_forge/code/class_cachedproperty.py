import collections.abc
import contextlib
import datetime
import functools
import inspect
import io
import os
import re
import socket
import sys
import threading
import types
import enum
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import netutils
from oslo_utils import reflection
from taskflow.types import failure
class cachedproperty(object):
    """A *thread-safe* descriptor property that is only evaluated once.

    This caching descriptor can be placed on instance methods to translate
    those methods into properties that will be cached in the instance (avoiding
    repeated attribute checking logic to do the equivalent).

    NOTE(harlowja): by default the property that will be saved will be under
    the decorated methods name prefixed with an underscore. For example if we
    were to attach this descriptor to an instance method 'get_thing(self)' the
    cached property would be stored under '_get_thing' in the self object
    after the first call to 'get_thing' occurs.
    """

    def __init__(self, fget=None, require_lock=True):
        if require_lock:
            self._lock = threading.RLock()
        else:
            self._lock = None
        if inspect.isfunction(fget):
            self._fget = fget
            self._attr_name = '_%s' % fget.__name__
            self.__doc__ = getattr(fget, '__doc__', None)
        else:
            self._attr_name = fget
            self._fget = None
            self.__doc__ = None

    def __call__(self, fget):
        self._fget = fget
        if not self._attr_name:
            self._attr_name = '_%s' % fget.__name__
        self.__doc__ = getattr(fget, '__doc__', None)
        return self

    def __set__(self, instance, value):
        raise AttributeError("can't set attribute")

    def __delete__(self, instance):
        raise AttributeError("can't delete attribute")

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if hasattr(instance, self._attr_name):
            return getattr(instance, self._attr_name)
        else:
            if self._lock is not None:
                self._lock.acquire()
            try:
                return getattr(instance, self._attr_name)
            except AttributeError:
                value = self._fget(instance)
                setattr(instance, self._attr_name, value)
                return value
            finally:
                if self._lock is not None:
                    self._lock.release()