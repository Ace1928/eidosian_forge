import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
def _borgify(self):
    """
        Create a new object that shares its state (i.e. its C{__dict__}) and
        type with this object, but does not share its identity.

        This is an instance of U{the Borg design pattern
        <https://code.activestate.com/recipes/66531/>} originally described by
        Alex Martelli, but unlike the example given there, this is not a
        replacement for a Singleton.  Instead, it is for lifecycle tracking
        (and distributed garbage collection).  The purpose of these separate
        objects is to have a separate object tracking each application-level
        reference to the root L{RemoteCache} object being tracked by the
        broker, and to have their C{__del__} methods be invoked.

        This may be achievable via a weak value dictionary to track the root
        L{RemoteCache} instances instead, but this implementation strategy
        predates the availability of weak references in Python.

        @return: The new instance.
        @rtype: C{self.__class__}
        """
    blank = _createBlank(self.__class__)
    blank.__dict__ = self.__dict__
    return blank