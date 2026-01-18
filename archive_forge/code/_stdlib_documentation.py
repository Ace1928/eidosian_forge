import logging as stdlibLogging
from typing import Mapping, Tuple
from zope.interface import implementer
from constantly import NamedConstant
from twisted.python.compat import currentframe
from ._format import formatEvent
from ._interfaces import ILogObserver, LogEvent
from ._levels import LogLevel
fromStdlibLogLevelMapping = _reverseLogLevelMapping()

        @param event: An event.
        