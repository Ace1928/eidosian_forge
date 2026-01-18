from collections.abc import Sequence
from sys import exc_info
from warnings import warn
from zope.interface import implementer
from twisted.internet.threads import blockingCallFromThread
from twisted.logger import Logger
from twisted.python.failure import Failure
from twisted.web.http import INTERNAL_SERVER_ERROR
from twisted.web.resource import IResource
from twisted.web.server import NOT_DONE_YET
class _InputStream:
    """
    File-like object instances of which are used as the value for the
    C{'wsgi.input'} key in the C{environ} dictionary passed to the application
    object.

    This only exists to make the handling of C{readline(-1)} consistent across
    different possible underlying file-like object implementations.  The other
    supported methods pass through directly to the wrapped object.
    """

    def __init__(self, input):
        """
        Initialize the instance.

        This is called in the I/O thread, not a WSGI application thread.
        """
        self._wrapped = input

    def read(self, size=None):
        """
        Pass through to the underlying C{read}.

        This is called in a WSGI application thread, not the I/O thread.
        """
        if size is None:
            return self._wrapped.read()
        return self._wrapped.read(size)

    def readline(self, size=None):
        """
        Pass through to the underlying C{readline}, with a size of C{-1} replaced
        with a size of L{None}.

        This is called in a WSGI application thread, not the I/O thread.
        """
        if size == -1 or size is None:
            return self._wrapped.readline()
        return self._wrapped.readline(size)

    def readlines(self, size=None):
        """
        Pass through to the underlying C{readlines}.

        This is called in a WSGI application thread, not the I/O thread.
        """
        if size is None:
            return self._wrapped.readlines()
        return self._wrapped.readlines(size)

    def __iter__(self):
        """
        Pass through to the underlying C{__iter__}.

        This is called in a WSGI application thread, not the I/O thread.
        """
        return iter(self._wrapped)