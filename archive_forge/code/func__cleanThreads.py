from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
def _cleanThreads(self):
    reactor = self._getReactor()
    if interfaces.IReactorThreads.providedBy(reactor):
        if reactor.threadpool is not None:
            reactor._stopThreadPool()