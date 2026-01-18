import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
class IPBRoot(Interface):
    """Factory for root Referenceable objects for PB servers."""

    def rootObject(broker):
        """Return root Referenceable for broker."""