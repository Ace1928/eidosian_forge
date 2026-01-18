from functools import partial
from typing import Dict, Iterable
from zope.interface import Interface, implementer
from constantly import NamedConstant, Names
from ._interfaces import ILogObserver, LogEvent
from ._levels import InvalidLogLevelError, LogLevel
from ._observer import bitbucketLogObserver

        Clears all log levels to the default.
        