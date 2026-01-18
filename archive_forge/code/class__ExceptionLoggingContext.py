import asyncio
import logging
import re
import types
from tornado.concurrent import (
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple
class _ExceptionLoggingContext(object):
    """Used with the ``with`` statement when calling delegate methods to
    log any exceptions with the given logger.  Any exceptions caught are
    converted to _QuietException
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def __enter__(self) -> None:
        pass

    def __exit__(self, typ: 'Optional[Type[BaseException]]', value: Optional[BaseException], tb: types.TracebackType) -> None:
        if value is not None:
            assert typ is not None
            self.logger.error('Uncaught exception', exc_info=(typ, value, tb))
            raise _QuietException