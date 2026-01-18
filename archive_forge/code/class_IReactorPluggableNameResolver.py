from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorPluggableNameResolver(Interface):
    """
    An L{IReactorPluggableNameResolver} is a reactor whose name resolver can be
    set to a user-supplied object.
    """
    nameResolver = Attribute('\n        Read-only attribute; the resolver installed with L{installResolver}.\n        An L{IHostnameResolver}.\n        ')

    def installNameResolver(resolver: IHostnameResolver) -> IHostnameResolver:
        """
        Set the internal resolver to use for name lookups.

        @param resolver: The new resolver to use.

        @return: The previously installed resolver.
        """