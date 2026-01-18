import select
import sys
from errno import EBADF, EINTR
from functools import partial
from queue import Empty, Queue
from threading import Thread
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, _NO_FILENO
from twisted.internet.selectreactor import _select
from twisted.python import failure, log, threadable
def _doReadOrWrite(self, selectable, method, dict):
    try:
        why = getattr(selectable, method)()
        handfn = getattr(selectable, 'fileno', None)
        if not handfn:
            why = _NO_FILENO
        elif handfn() == -1:
            why = _NO_FILEDESC
    except BaseException:
        why = sys.exc_info()[1]
        log.err()
    if why:
        self._disconnectSelectable(selectable, why, method == 'doRead')