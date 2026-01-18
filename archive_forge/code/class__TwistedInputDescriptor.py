from __future__ import annotations
import functools
import logging
import sys
import typing
from twisted.internet.abstract import FileDescriptor
from twisted.internet.error import AlreadyCalled, AlreadyCancelled
from .abstract_loop import EventLoop, ExitMainLoop
class _TwistedInputDescriptor(FileDescriptor):

    def __init__(self, reactor: ReactorBase, fd: int, cb: Callable[[], typing.Any]) -> None:
        self._fileno = fd
        self.cb = cb
        super().__init__(reactor)

    def fileno(self) -> int:
        return self._fileno

    def doRead(self):
        return self.cb()

    def getHost(self):
        raise NotImplementedError('No network operation expected')

    def getPeer(self):
        raise NotImplementedError('No network operation expected')

    def writeSomeData(self, data: bytes) -> None:
        raise NotImplementedError('Reduced functionality: read-only')