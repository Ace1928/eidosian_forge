import re
from zope.interface import implementer
from twisted.internet.defer import (
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import (
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
def _ensureValidURI(uri):
    """
    A valid URI cannot contain control characters (i.e., characters
    between 0-32, inclusive and 127) or non-ASCII characters (i.e.,
    characters with values between 128-255, inclusive).

    @param uri: the URI to check
    @type uri: L{bytes}

    @return: the URI if it is valid
    @rtype: L{bytes}

    @raise ValueError: if the URI is not valid

    @see: U{https://tools.ietf.org/html/rfc3986#section-3.3},
        U{https://tools.ietf.org/html/rfc3986#appendix-A},
        U{https://tools.ietf.org/html/rfc5234#appendix-B.1}
    """
    if _VALID_URI.match(uri):
        return uri
    raise ValueError(f'Invalid URI {uri!r}')