import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def _cdef(self, csource, override=False, **options):
    if not isinstance(csource, str):
        if not isinstance(csource, basestring):
            raise TypeError('cdef() argument must be a string')
        csource = csource.encode('ascii')
    with self._lock:
        self._cdef_version = object()
        self._parser.parse(csource, override=override, **options)
        self._cdefsources.append(csource)
        if override:
            for cache in self._function_caches:
                cache.clear()
        finishlist = self._parser._recomplete
        if finishlist:
            self._parser._recomplete = []
            for tp in finishlist:
                tp.finish_backend_type(self, finishlist)