from eventlet.patcher import slurp_properties
import sys
from eventlet import greenio, hubs
from eventlet.greenio import (
from eventlet.hubs import trampoline, IOClosed
from eventlet.support import get_errno, PY33
from contextlib import contextmanager
class GreenSSLContext(_original_sslcontext):
    __slots__ = ()

    def wrap_socket(self, sock, *a, **kw):
        return GreenSSLSocket(sock, *a, _context=self, **kw)
    if hasattr(_original_sslcontext.options, 'setter'):

        @_original_sslcontext.options.setter
        def options(self, value):
            super(_original_sslcontext, _original_sslcontext).options.__set__(self, value)

        @_original_sslcontext.verify_flags.setter
        def verify_flags(self, value):
            super(_original_sslcontext, _original_sslcontext).verify_flags.__set__(self, value)

        @_original_sslcontext.verify_mode.setter
        def verify_mode(self, value):
            super(_original_sslcontext, _original_sslcontext).verify_mode.__set__(self, value)
        if hasattr(_original_sslcontext, 'maximum_version'):

            @_original_sslcontext.maximum_version.setter
            def maximum_version(self, value):
                super(_original_sslcontext, _original_sslcontext).maximum_version.__set__(self, value)
        if hasattr(_original_sslcontext, 'minimum_version'):

            @_original_sslcontext.minimum_version.setter
            def minimum_version(self, value):
                super(_original_sslcontext, _original_sslcontext).minimum_version.__set__(self, value)