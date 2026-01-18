import threading
import sys
from paste.util import filemixin
class StdinCatcher(filemixin.FileMixin):

    def __init__(self, default=None, factory=None, paramwriter=None):
        assert len(filter(lambda x: x is not None, [default, factory, paramwriter])) <= 1, 'You can only provide one of default, factory, or paramwriter'
        if default:
            self._defaultfunc = self._readdefault
        elif factory:
            self._defaultfunc = self._readfactory
        elif paramwriter:
            self._defaultfunc = self._readparam
        else:
            self._defaultfunc = self._readerror
        self._default = default
        self._factory = factory
        self._paramwriter = paramwriter
        self._catchers = {}

    def read(self, size=None, currentThread=threading.current_thread):
        name = currentThread().name
        catchers = self._catchers
        if not catchers.has_key(name):
            return self._defaultfunc(name, size)
        else:
            catcher = catchers[name]
            return catcher.read(size)

    def _readdefault(self, name, size):
        self._default.read(size)

    def _readfactory(self, name, size):
        self._factory(name).read(size)

    def _readparam(self, name, size):
        self._paramreader(name, size)

    def _readerror(self, name, size):
        assert False, 'There is no StdinCatcher output stream for the thread %r' % name

    def register(self, catcher, name=None, currentThread=threading.current_thread):
        if name is None:
            name = currentThread().name
        self._catchers[name] = catcher

    def deregister(self, catcher, name=None, currentThread=threading.current_thread):
        if name is None:
            name = currentThread().name
        assert self._catchers.has_key(name), 'There is no StdinCatcher catcher for the thread %r' % name
        del self._catchers[name]