from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
class IRenderable(Interface):
    """
    An L{IRenderable} is an object that may be rendered by the
    L{twisted.web.template} templating system.
    """

    def lookupRenderMethod(name: str) -> Callable[[Optional[IRequest], 'Tag'], 'Flattenable']:
        """
        Look up and return the render method associated with the given name.

        @param name: The value of a render directive encountered in the
            document returned by a call to L{IRenderable.render}.

        @return: A two-argument callable which will be invoked with the request
            being responded to and the tag object on which the render directive
            was encountered.
        """

    def render(request: Optional[IRequest]) -> 'Flattenable':
        """
        Get the document for this L{IRenderable}.

        @param request: The request in response to which this method is being
            invoked.

        @return: An object which can be flattened.
        """