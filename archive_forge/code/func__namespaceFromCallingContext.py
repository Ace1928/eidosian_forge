from time import time
from typing import Any, Optional, cast
from twisted.python.compat import currentframe
from twisted.python.failure import Failure
from ._interfaces import ILogObserver, LogTrace
from ._levels import InvalidLogLevelError, LogLevel
@staticmethod
def _namespaceFromCallingContext() -> str:
    """
        Derive a namespace from the module containing the caller's caller.

        @return: the fully qualified python name of a module.
        """
    try:
        return cast(str, currentframe(2).f_globals['__name__'])
    except KeyError:
        return '<unknown>'