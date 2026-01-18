import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
class RemoteCopy(Unjellyable):
    """I am a remote copy of a Copyable object.

    When the state from a L{Copyable} object is received, an instance will
    be created based on the copy tags table (see setUnjellyableForClass) and
    sent the L{setCopyableState} message.  I provide a reasonable default
    implementation of that message; subclass me if you wish to serve as
    a copier for remote data.

    NOTE: copiers are invoked with no arguments.  Do not implement a
    constructor which requires args in a subclass of L{RemoteCopy}!
    """

    def setCopyableState(self, state):
        """I will be invoked with the state to copy locally.

        'state' is the data returned from the remote object's
        'getStateToCopyFor' method, which will often be the remote
        object's dictionary (or a filtered approximation of it depending
        on my peer's perspective).
        """
        if not state:
            state = {}
        state = {x.decode('utf8') if isinstance(x, bytes) else x: y for x, y in state.items()}
        self.__dict__ = state

    def unjellyFor(self, unjellier, jellyList):
        if unjellier.invoker is None:
            return setInstanceState(self, unjellier, jellyList)
        self.setCopyableState(unjellier.unjelly(jellyList[1]))
        return self