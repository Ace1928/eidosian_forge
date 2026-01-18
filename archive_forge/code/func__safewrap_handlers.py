import logging
import numbers
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from typing import AnyStr, Sequence  # noqa
from kombu.log import LOG_LEVELS
from kombu.log import get_logger as _get_logger
from kombu.utils.encoding import safe_str
from .term import colored
def _safewrap_handlers(self):

    def wrap_handler(handler):

        class WithSafeHandleError(logging.Handler):

            def handleError(self, record):
                try:
                    traceback.print_exc(None, sys.__stderr__)
                except OSError:
                    pass
        handler.handleError = WithSafeHandleError().handleError
    return [wrap_handler(h) for h in self.logger.handlers]