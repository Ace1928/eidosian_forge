import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
class Finalize(object):
    """
    Class which supports object finalization using weakrefs
    """

    def __init__(self, obj, callback, args=(), kwargs=None, exitpriority=None):
        if exitpriority is not None and (not isinstance(exitpriority, int)):
            raise TypeError('Exitpriority ({0!r}) must be None or int, not {1!s}'.format(exitpriority, type(exitpriority)))
        if obj is not None:
            self._weakref = weakref.ref(obj, self)
        elif exitpriority is None:
            raise ValueError('Without object, exitpriority cannot be None')
        self._callback = callback
        self._args = args
        self._kwargs = kwargs or {}
        self._key = (exitpriority, next(_finalizer_counter))
        self._pid = os.getpid()
        _finalizer_registry[self._key] = self

    def __call__(self, wr=None, _finalizer_registry=_finalizer_registry, sub_debug=sub_debug, getpid=os.getpid):
        """
        Run the callback unless it has already been called or cancelled
        """
        try:
            del _finalizer_registry[self._key]
        except KeyError:
            sub_debug('finalizer no longer registered')
        else:
            if self._pid != getpid():
                sub_debug('finalizer ignored because different process')
                res = None
            else:
                sub_debug('finalizer calling %s with args %s and kwargs %s', self._callback, self._args, self._kwargs)
                res = self._callback(*self._args, **self._kwargs)
            self._weakref = self._callback = self._args = self._kwargs = self._key = None
            return res

    def cancel(self):
        """
        Cancel finalization of the object
        """
        try:
            del _finalizer_registry[self._key]
        except KeyError:
            pass
        else:
            self._weakref = self._callback = self._args = self._kwargs = self._key = None

    def still_active(self):
        """
        Return whether this finalizer is still waiting to invoke callback
        """
        return self._key in _finalizer_registry

    def __repr__(self):
        try:
            obj = self._weakref()
        except (AttributeError, TypeError):
            obj = None
        if obj is None:
            return '<%s object, dead>' % self.__class__.__name__
        x = '<%s object, callback=%s' % (self.__class__.__name__, getattr(self._callback, '__name__', self._callback))
        if self._args:
            x += ', args=' + str(self._args)
        if self._kwargs:
            x += ', kwargs=' + str(self._kwargs)
        if self._key[0] is not None:
            x += ', exitpriority=' + str(self._key[0])
        return x + '>'