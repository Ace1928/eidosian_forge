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
def _doSelectInThread(self, timeout):
    """Run one iteration of the I/O monitor loop.

        This will run all selectables who had input or output readiness
        waiting for them.
        """
    reads = self.reads
    writes = self.writes
    while 1:
        try:
            r, w, ignored = _select(reads.keys(), writes.keys(), [], timeout)
            break
        except ValueError:
            log.err()
            self._preenDescriptorsInThread()
        except TypeError:
            log.err()
            self._preenDescriptorsInThread()
        except OSError as se:
            if se.args[0] in (0, 2):
                if not reads and (not writes):
                    return
                else:
                    raise
            elif se.args[0] == EINTR:
                return
            elif se.args[0] == EBADF:
                self._preenDescriptorsInThread()
            else:
                raise
    self._sendToMain('Notify', r, w)