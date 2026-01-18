import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
class Viewable(Serializable):
    """I will be converted to a L{ViewPoint} when passed to or returned from a remote method.

    The beginning of a peer's interaction with a PB Service is always
    through a perspective.  However, if a C{perspective_xxx} method returns
    a Viewable, it will be serialized to the peer as a response to that
    method.
    """

    def jellyFor(self, jellier):
        """Serialize a L{ViewPoint} for me and the perspective of the given broker."""
        return ViewPoint(jellier.invoker.serializingPerspective, self).jellyFor(jellier)