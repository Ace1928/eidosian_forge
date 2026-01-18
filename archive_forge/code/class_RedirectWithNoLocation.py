from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class RedirectWithNoLocation(Error):
    """
    Exception passed to L{ResponseFailed} if we got a redirect without a
    C{Location} header field.

    @type uri: L{bytes}
    @ivar uri: The URI which failed to give a proper location header
        field.

    @since: 11.1
    """
    message: bytes
    uri: bytes

    def __init__(self, code: Union[bytes, int], message: bytes, uri: bytes) -> None:
        """
        Initializes a page redirect exception when no location is given.

        @type code: L{bytes}
        @param code: Refers to an HTTP status code, for example
            C{http.NOT_FOUND}. If no C{message} is given, C{code} is mapped to
            a descriptive string that is used instead.

        @type message: L{bytes}
        @param message: A short error message.

        @type uri: L{bytes}
        @param uri: The URI which failed to give a proper location header
            field.
        """
        Error.__init__(self, code, message)
        self.message = self.message + b' to ' + uri
        self.uri = uri