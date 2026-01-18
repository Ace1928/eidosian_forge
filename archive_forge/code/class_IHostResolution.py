from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IHostResolution(Interface):
    """
    An L{IHostResolution} represents represents an in-progress recursive query
    for a DNS name.

    @since: Twisted 17.1.0
    """
    name = Attribute('\n        L{unicode}; the name of the host being resolved.\n        ')

    def cancel() -> None:
        """
        Stop the hostname resolution in progress.
        """