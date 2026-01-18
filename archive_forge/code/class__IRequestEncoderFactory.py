from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
class _IRequestEncoderFactory(Interface):
    """
    A factory for returing L{_IRequestEncoder} instances.

    @since: 12.3
    """

    def encoderForRequest(request):
        """
        If applicable, returns a L{_IRequestEncoder} instance which will encode
        the request.
        """