import errno
import sys
from asyncio import AbstractEventLoop, get_event_loop
from typing import Dict, Optional, Type
from zope.interface import implementer
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import (
from twisted.logger import Logger
from twisted.python.log import callWithLogger
def _readOrWrite(self, selectable, read):
    method = selectable.doRead if read else selectable.doWrite
    if selectable.fileno() == -1:
        self._disconnectSelectable(selectable, _NO_FILEDESC, read)
        return
    try:
        why = method()
    except Exception as e:
        why = e
        self._log.failure(None)
    if why:
        self._disconnectSelectable(selectable, why, read)