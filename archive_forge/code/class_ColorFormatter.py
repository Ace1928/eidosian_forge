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
class ColorFormatter(logging.Formatter):
    """Logging formatter that adds colors based on severity."""
    COLORS = colored().names
    colors = {'DEBUG': COLORS['blue'], 'WARNING': COLORS['yellow'], 'ERROR': COLORS['red'], 'CRITICAL': COLORS['magenta']}

    def __init__(self, fmt=None, use_color=True):
        super().__init__(fmt)
        self.use_color = use_color

    def formatException(self, ei):
        if ei and (not isinstance(ei, tuple)):
            ei = sys.exc_info()
        r = super().formatException(ei)
        return r

    def format(self, record):
        msg = super().format(record)
        color = self.colors.get(record.levelname)
        einfo = sys.exc_info() if record.exc_info == 1 else record.exc_info
        if color and self.use_color:
            try:
                try:
                    if isinstance(msg, str):
                        return str(color(safe_str(msg)))
                    return safe_str(color(msg))
                except UnicodeDecodeError:
                    return safe_str(msg)
            except Exception as exc:
                prev_msg, record.exc_info, record.msg = (record.msg, 1, '<Unrepresentable {!r}: {!r}>'.format(type(msg), exc))
                try:
                    return super().format(record)
                finally:
                    record.msg, record.exc_info = (prev_msg, einfo)
        else:
            return safe_str(msg)