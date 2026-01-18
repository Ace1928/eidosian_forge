from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
class IAgent(Interface):
    """
    An agent makes HTTP requests.

    The way in which requests are issued is left up to each implementation.
    Some may issue them directly to the server indicated by the net location
    portion of the request URL.  Others may use a proxy specified by system
    configuration.

    Processing of responses is also left very widely specified.  An
    implementation may perform no special handling of responses, or it may
    implement redirect following or content negotiation, it may implement a
    cookie store or automatically respond to authentication challenges.  It may
    implement many other unforeseen behaviors as well.

    It is also intended that L{IAgent} implementations be composable.  An
    implementation which provides cookie handling features should re-use an
    implementation that provides connection pooling and this combination could
    be used by an implementation which adds content negotiation functionality.
    Some implementations will be completely self-contained, such as those which
    actually perform the network operations to send and receive requests, but
    most or all other implementations should implement a small number of new
    features (perhaps one new feature) and delegate the rest of the
    request/response machinery to another implementation.

    This allows for great flexibility in the behavior an L{IAgent} will
    provide.  For example, an L{IAgent} with web browser-like behavior could be
    obtained by combining a number of (hypothetical) implementations::

        baseAgent = Agent(reactor)
        decode = ContentDecoderAgent(baseAgent, [(b"gzip", GzipDecoder())])
        cookie = CookieAgent(decode, diskStore.cookie)
        authenticate = AuthenticateAgent(
            cookie, [diskStore.credentials, GtkAuthInterface()])
        cache = CacheAgent(authenticate, diskStore.cache)
        redirect = BrowserLikeRedirectAgent(cache, limit=10)

        doSomeRequests(cache)
    """

    def request(method: bytes, uri: bytes, headers: Optional[Headers]=None, bodyProducer: Optional[IBodyProducer]=None) -> Deferred[IResponse]:
        """
        Request the resource at the given location.

        @param method: The request method to use, such as C{b"GET"}, C{b"HEAD"},
            C{b"PUT"}, C{b"POST"}, etc.

        @param uri: The location of the resource to request.  This should be an
            absolute URI but some implementations may support relative URIs
            (with absolute or relative paths).  I{HTTP} and I{HTTPS} are the
            schemes most likely to be supported but others may be as well.

        @param headers: The headers to send with the request (or L{None} to
            send no extra headers).  An implementation may add its own headers
            to this (for example for client identification or content
            negotiation).

        @param bodyProducer: An object which can generate bytes to make up the
            body of this request (for example, the properly encoded contents of
            a file for a file upload).  Or, L{None} if the request is to have
            no body.

        @return: A L{Deferred} that fires with an L{IResponse} provider when
            the header of the response has been received (regardless of the
            response status code) or with a L{Failure} if there is any problem
            which prevents that response from being received (including
            problems that prevent the request from being sent).
        """