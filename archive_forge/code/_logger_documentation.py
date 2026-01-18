from time import time
from typing import Any, Optional, cast
from twisted.python.compat import currentframe
from twisted.python.failure import Failure
from ._interfaces import ILogObserver, LogTrace
from ._levels import InvalidLogLevelError, LogLevel

        Emit a log event at log level L{LogLevel.critical}.

        @param format: a message format using new-style (PEP 3101) formatting.
            The logging event (which is a L{dict}) is used to render this
            format string.

        @param kwargs: additional key/value pairs to include in the event.
            Note that values which are later mutated may result in
            non-deterministic behavior from observers that schedule work for
            later execution.
        