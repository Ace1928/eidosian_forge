from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
class IAccessLogFormatter(Interface):
    """
    An object which can represent an HTTP request as a line of text for
    inclusion in an access log file.
    """

    def __call__(timestamp, request):
        """
        Generate a line for the access log.

        @param timestamp: The time at which the request was completed in the
            standard format for access logs.
        @type timestamp: L{unicode}

        @param request: The request object about which to log.
        @type request: L{twisted.web.server.Request}

        @return: One line describing the request without a trailing newline.
        @rtype: L{unicode}
        """