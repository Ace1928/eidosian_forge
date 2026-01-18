from __future__ import annotations
from threading import Thread, current_thread
from typing import Any, Callable, List, Optional, TypeVar
from typing_extensions import ParamSpec, Protocol, TypedDict
from twisted._threads import pool as _pool
from twisted.python import context, log
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.python.versions import Version
def _generateName(self) -> str:
    """
        Generate a name for a new pool thread.

        @return: A distinctive name for the thread.
        @rtype: native L{str}
        """
    return f'PoolThread-{self.name or id(self)}-{self.workers}'