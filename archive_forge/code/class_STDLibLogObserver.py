import logging as stdlibLogging
from typing import Mapping, Tuple
from zope.interface import implementer
from constantly import NamedConstant
from twisted.python.compat import currentframe
from ._format import formatEvent
from ._interfaces import ILogObserver, LogEvent
from ._levels import LogLevel
fromStdlibLogLevelMapping = _reverseLogLevelMapping()
@implementer(ILogObserver)
class STDLibLogObserver:
    """
    Log observer that writes to the python standard library's C{logging}
    module.

    @note: Warning: specific logging configurations (example: network) can lead
        to this observer blocking.  Nothing is done here to prevent that, so be
        sure to not to configure the standard library logging module to block
        when used in conjunction with this module: code within Twisted, such as
        twisted.web, assumes that logging does not block.

    @cvar defaultStackDepth: This is the default number of frames that it takes
        to get from L{STDLibLogObserver} through the logging module, plus one;
        in other words, the number of frames if you were to call a
        L{STDLibLogObserver} directly.  This is useful to use as an offset for
        the C{stackDepth} parameter to C{__init__}, to add frames for other
        publishers.
    """
    defaultStackDepth = 4

    def __init__(self, name: str='twisted', stackDepth: int=defaultStackDepth) -> None:
        """
        @param name: logger identifier.
        @param stackDepth: The depth of the stack to investigate for caller
            metadata.
        """
        self.logger = stdlibLogging.getLogger(name)
        self.logger.findCaller = self._findCaller
        self.stackDepth = stackDepth

    def _findCaller(self, stackInfo: bool=False, stackLevel: int=1) -> Tuple[str, int, str, None]:
        """
        Based on the stack depth passed to this L{STDLibLogObserver}, identify
        the calling function.

        @param stackInfo: Whether or not to construct stack information.
            (Currently ignored.)
        @param stackLevel: The number of stack frames to skip when determining
            the caller (currently ignored; use stackDepth on the instance).

        @return: Depending on Python version, either a 3-tuple of (filename,
            lineno, name) or a 4-tuple of that plus stack information.
        """
        f = currentframe(self.stackDepth)
        co = f.f_code
        extra = (None,)
        return (co.co_filename, f.f_lineno, co.co_name) + extra

    def __call__(self, event: LogEvent) -> None:
        """
        Format an event and bridge it to Python logging.
        """
        level = event.get('log_level', LogLevel.info)
        failure = event.get('log_failure')
        if failure is None:
            excInfo = None
        else:
            excInfo = (failure.type, failure.value, failure.getTracebackObject())
        stdlibLevel = toStdlibLogLevelMapping.get(level, stdlibLogging.INFO)
        self.logger.log(stdlibLevel, StringifiableFromEvent(event), exc_info=excInfo)