from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorPluggableResolver(Interface):
    """
    An L{IReactorPluggableResolver} is a reactor which can be customized with
    an L{IResolverSimple}.  This is a fairly limited interface, that supports
    only IPv4; you should use L{IReactorPluggableNameResolver} instead.

    @see: L{IReactorPluggableNameResolver}
    """

    def installResolver(resolver: IResolverSimple) -> IResolverSimple:
        """
        Set the internal resolver to use to for name lookups.

        @param resolver: The new resolver to use.

        @return: The previously installed resolver.
        """