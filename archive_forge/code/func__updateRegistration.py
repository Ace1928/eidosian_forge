import errno
from select import (
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
def _updateRegistration(self, fd):
    """Register/unregister an fd with the poller."""
    try:
        self._poller.unregister(fd)
    except KeyError:
        pass
    mask = 0
    if fd in self._reads:
        mask = mask | POLLIN
    if fd in self._writes:
        mask = mask | POLLOUT
    if mask != 0:
        self._poller.register(fd, mask)
    elif fd in self._selectables:
        del self._selectables[fd]