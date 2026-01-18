from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IStreamServerEndpointStringParser(Interface):
    """
    An L{IStreamServerEndpointStringParser} is like an
    L{IStreamClientEndpointStringParserWithReactor}, except for
    L{IStreamServerEndpoint}s instead of clients.  It integrates with
    L{endpoints.serverFromString} in much the same way.
    """
    prefix = Attribute('\n        A C{str}, the description prefix to respond to.  For example, an\n        L{IStreamServerEndpointStringParser} plugin which had C{"foo"} for its\n        C{prefix} attribute would be called for endpoint descriptions like\n        C{"foo:bar:baz"} or C{"foo:"}.\n        ')

    def parseStreamServer(reactor: IReactorCore, *args: object, **kwargs: object) -> IStreamServerEndpoint:
        """
        Parse a stream server endpoint from a reactor and string-only arguments
        and keyword arguments.

        @see: L{IStreamClientEndpointStringParserWithReactor.parseStreamClient}

        @return: a stream server endpoint
        """