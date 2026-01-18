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
def _preenDescriptorsInThread(self):
    log.msg('Malformed file descriptor found.  Preening lists.')
    readers = self.reads.keys()
    writers = self.writes.keys()
    self.reads.clear()
    self.writes.clear()
    for selDict, selList in ((self.reads, readers), (self.writes, writers)):
        for selectable in selList:
            try:
                select.select([selectable], [selectable], [selectable], 0)
            except BaseException:
                log.msg('bad descriptor %s' % selectable)
            else:
                selDict[selectable] = 1