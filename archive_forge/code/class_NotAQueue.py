from __future__ import annotations
from threading import Thread, current_thread
from typing import Any, Callable, List, Optional, TypeVar
from typing_extensions import ParamSpec, Protocol, TypedDict
from twisted._threads import pool as _pool
from twisted.python import context, log
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.python.versions import Version
class NotAQueue:

    def qsize(q) -> int:
        """
                Pretend to be a Python threading Queue and return the
                number of as-yet-unconsumed tasks.

                @return: the amount of backlogged work not yet dispatched to a
                    worker.
                @rtype: L{int}
                """
        return self._team.statistics().backloggedWorkCount