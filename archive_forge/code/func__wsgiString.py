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
def _wsgiString(string):
    """
        Convert C{string} to a WSGI "bytes-as-unicode" string.

        If it's a byte string, decode as ISO-8859-1. If it's a Unicode string,
        round-trip it to bytes and back using ISO-8859-1 as the encoding.

        @type string: C{str} or C{bytes}
        @rtype: C{str}

        @raise UnicodeEncodeError: If C{string} contains non-ISO-8859-1 chars.
        """
    if isinstance(string, str):
        return string.encode('iso-8859-1').decode('iso-8859-1')
    else:
        return string.decode('iso-8859-1')