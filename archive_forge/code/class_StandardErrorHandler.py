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
class StandardErrorHandler(logging.StreamHandler):
    """
    A :class:`~logging.StreamHandler` that gets the value of :data:`sys.stderr` for each log message.

    The :class:`StandardErrorHandler` class enables `monkey patching of
    sys.stderr <https://github.com/xolox/python-coloredlogs/pull/31>`_. It's
    basically the same as the ``logging._StderrHandler`` class present in
    Python 3 but it will be available regardless of Python version. This
    handler is used by :func:`coloredlogs.install()` to improve compatibility
    with the Python standard library.
    """

    def __init__(self, level=logging.NOTSET):
        """Initialize a :class:`StandardErrorHandler` object."""
        logging.Handler.__init__(self, level)

    @property
    def stream(self):
        """Get the value of :data:`sys.stderr` (a file-like object)."""
        return sys.stderr