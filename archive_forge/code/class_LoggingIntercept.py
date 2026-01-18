import inspect
import io
import logging
import re
import sys
import textwrap
from pyomo.version.info import releaselevel
from pyomo.common.deprecation import deprecated
from pyomo.common.fileutils import PYOMO_ROOT_DIR
from pyomo.common.formatting import wrap_reStructuredText
class LoggingIntercept(object):
    """Context manager for intercepting messages sent to a log stream

    This class is designed to enable easy testing of log messages.

    The LoggingIntercept context manager will intercept messages sent to
    a log stream matching a specified level and send the messages to the
    specified output stream.  Other handlers registered to the target
    logger will be temporarily removed and the logger will be set not to
    propagate messages up to higher-level loggers.

    Parameters
    ----------
    output: io.TextIOBase
        the file stream to send log messages to
    module: str
        the target logger name to intercept
    level: int
        the logging level to intercept
    formatter: logging.Formatter
        the formatter to use when rendering the log messages.  If not
        specified, uses `'%(message)s'`

    Examples:
        >>> import io, logging
        >>> from pyomo.common.log import LoggingIntercept
        >>> buf = io.StringIO()
        >>> with LoggingIntercept(buf, 'pyomo.core', logging.WARNING):
        ...     logging.getLogger('pyomo.core').warning('a simple message')
        >>> buf.getvalue()

    """

    def __init__(self, output=None, module=None, level=logging.WARNING, formatter=None):
        self.handler = None
        self.output = output
        self.module = module
        self._level = level
        if formatter is None:
            formatter = logging.Formatter('%(message)s')
        self._formatter = formatter
        self._save = None

    def __enter__(self):
        output = self.output
        if output is None:
            output = io.StringIO()
        assert self.handler is None
        self.handler = logging.StreamHandler(output)
        self.handler.setFormatter(self._formatter)
        self.handler.setLevel(self._level)
        logger = logging.getLogger(self.module)
        self._save = (logger.level, logger.propagate, logger.handlers)
        logger.handlers = []
        logger.propagate = 0
        logger.setLevel(self.handler.level)
        logger.addHandler(self.handler)
        return output

    def __exit__(self, et, ev, tb):
        logger = logging.getLogger(self.module)
        logger.removeHandler(self.handler)
        self.handler = None
        logger.setLevel(self._save[0])
        logger.propagate = self._save[1]
        for h in self._save[2]:
            logger.handlers.append(h)