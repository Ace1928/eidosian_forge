from queue import Queue
from threading import Lock, Thread, local as LocalStorage
from typing import Callable, Optional
from typing_extensions import Protocol
from twisted.python.log import err
from ._ithreads import IWorker
from ._team import Team
from ._threadworker import LockWorker, ThreadWorker
class _ThreadFactory(Protocol):

    def __call__(self, *, target: Callable[..., object]) -> Thread:
        ...