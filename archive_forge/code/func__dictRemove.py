import errno
from select import (
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
def _dictRemove(self, selectable, mdict):
    try:
        fd = selectable.fileno()
        mdict[fd]
    except BaseException:
        for fd, fdes in self._selectables.items():
            if selectable is fdes:
                break
        else:
            return
    if fd in mdict:
        del mdict[fd]
        self._updateRegistration(fd)