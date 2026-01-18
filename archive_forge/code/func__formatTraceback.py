from __future__ import annotations
from datetime import datetime as DateTime
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast
from constantly import NamedConstant
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.python.failure import Failure
from twisted.python.reflect import safe_repr
from ._flatten import aFormatter, flatFormat
from ._interfaces import LogEvent
def _formatTraceback(failure: Failure) -> str:
    """
    Format a failure traceback, assuming UTF-8 and using a replacement
    strategy for errors.  Every effort is made to provide a usable
    traceback, but should not that not be possible, a message and the
    captured exception are logged.

    @param failure: The failure to retrieve a traceback from.

    @return: The formatted traceback.
    """
    try:
        traceback = failure.getTraceback()
    except BaseException as e:
        traceback = '(UNABLE TO OBTAIN TRACEBACK FROM EVENT):' + str(e)
    return traceback