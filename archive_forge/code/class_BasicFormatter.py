import collections
import logging
import os
import re
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import coerce_string, is_string, on_windows
from humanfriendly.terminal import ANSI_COLOR_CODES, ansi_wrap, enable_ansi_support, terminal_supports_colors
from humanfriendly.text import format, split
class BasicFormatter(logging.Formatter):
    """
    Log :class:`~logging.Formatter` that supports ``%f`` for millisecond formatting.

    This class extends :class:`~logging.Formatter` to enable the use of ``%f``
    for millisecond formatting in date/time strings, to allow for the type of
    flexibility requested in issue `#45`_.

    .. _#45: https://github.com/xolox/python-coloredlogs/issues/45
    """

    def formatTime(self, record, datefmt=None):
        """
        Format the date/time of a log record.

        :param record: A :class:`~logging.LogRecord` object.
        :param datefmt: A date/time format string (defaults to :data:`DEFAULT_DATE_FORMAT`).
        :returns: The formatted date/time (a string).

        This method overrides :func:`~logging.Formatter.formatTime()` to set
        `datefmt` to :data:`DEFAULT_DATE_FORMAT` when the caller hasn't
        specified a date format.

        When `datefmt` contains the token ``%f`` it will be replaced by the
        value of ``%(msecs)03d`` (refer to issue `#45`_ for use cases).
        """
        datefmt = datefmt or DEFAULT_DATE_FORMAT
        if '%f' in datefmt:
            datefmt = datefmt.replace('%f', '%03d' % record.msecs)
        return logging.Formatter.formatTime(self, record, datefmt)