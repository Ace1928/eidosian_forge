from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
class INonQueuedRequestFactory(Interface):
    """
    A factory of L{IRequest} objects that does not take a ``queued`` parameter.
    """

    def __call__(channel):
        """
        Create an L{IRequest} that is operating on the given channel. There
        must only be one L{IRequest} object processing at any given time on a
        channel.

        @param channel: A L{twisted.web.http.HTTPChannel} object.
        @type channel: L{twisted.web.http.HTTPChannel}

        @return: A request object.
        @rtype: L{IRequest}
        """