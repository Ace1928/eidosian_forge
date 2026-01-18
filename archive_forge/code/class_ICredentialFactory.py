from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
class ICredentialFactory(Interface):
    """
    A credential factory defines a way to generate a particular kind of
    authentication challenge and a way to interpret the responses to these
    challenges.  It creates
    L{ICredentials<twisted.cred.credentials.ICredentials>} providers from
    responses.  These objects will be used with L{twisted.cred} to authenticate
    an authorize requests.
    """
    scheme = Attribute("A L{str} giving the name of the authentication scheme with which this factory is associated.  For example, C{'basic'} or C{'digest'}.")

    def getChallenge(request):
        """
        Generate a new challenge to be sent to a client.

        @type request: L{twisted.web.http.Request}
        @param request: The request the response to which this challenge will
            be included.

        @rtype: L{dict}
        @return: A mapping from L{str} challenge fields to associated L{str}
            values.
        """

    def decode(response, request):
        """
        Create a credentials object from the given response.

        @type response: L{str}
        @param response: scheme specific response string

        @type request: L{twisted.web.http.Request}
        @param request: The request being processed (from which the response
            was taken).

        @raise twisted.cred.error.LoginFailed: If the response is invalid.

        @rtype: L{twisted.cred.credentials.ICredentials} provider
        @return: The credentials represented by the given response.
        """