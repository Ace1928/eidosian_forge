import socket
import sys
from typing import Sequence
from zope.interface import classImplements, implementer
from twisted.internet import error, tcp, udp
from twisted.internet.base import ReactorBase
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform, platformType
from ._signals import (
def _removeAll(self, readers, writers):
    """
        Remove all readers and writers, and list of removed L{IReadDescriptor}s
        and L{IWriteDescriptor}s.

        Meant for calling from subclasses, to implement removeAll, like::

          def removeAll(self):
              return self._removeAll(self._reads, self._writes)

        where C{self._reads} and C{self._writes} are iterables.
        """
    removedReaders = set(readers) - self._internalReaders
    for reader in removedReaders:
        self.removeReader(reader)
    removedWriters = set(writers)
    for writer in removedWriters:
        self.removeWriter(writer)
    return list(removedReaders | removedWriters)