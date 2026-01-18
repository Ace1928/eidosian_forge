import asyncio
import enum
import sys
import warnings
from types import TracebackType
from typing import Optional, Type
Set deadline to absolute value.

        deadline argument points on the time in the same clock system
        as loop.time().

        If new deadline is in the past the timeout is raised immediately.

        Please note: it is not POSIX time but a time with
        undefined starting base, e.g. the time of the system power on.
        