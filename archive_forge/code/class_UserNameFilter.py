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
class UserNameFilter(logging.Filter):
    """
    Log filter to enable the ``%(username)s`` format.

    Python's :mod:`logging` module doesn't expose the username of the currently
    logged in user as requested in `#76`_. Given that :class:`HostNameFilter`
    and :class:`ProgramNameFilter` are already provided by `coloredlogs` it
    made sense to provide :class:`UserNameFilter` as well.

    Refer to :class:`HostNameFilter` for an example of how to manually install
    these log filters.

    .. _#76: https://github.com/xolox/python-coloredlogs/issues/76
    """

    @classmethod
    def install(cls, handler, fmt, username=None, style=DEFAULT_FORMAT_STYLE):
        """
        Install the :class:`UserNameFilter` (only if needed).

        :param fmt: The log format string to check for ``%(username)``.
        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`).
        :param handler: The logging handler on which to install the filter.
        :param username: Refer to :func:`__init__()`.

        If `fmt` is given the filter will only be installed if `fmt` uses the
        ``username`` field. If `fmt` is not given the filter is installed
        unconditionally.
        """
        if fmt:
            parser = FormatStringParser(style=style)
            if not parser.contains_field(fmt, 'username'):
                return
        handler.addFilter(cls(username))

    def __init__(self, username=None):
        """
        Initialize a :class:`UserNameFilter` object.

        :param username: The username to use (defaults to the
                         result of :func:`find_username()`).
        """
        self.username = username or find_username()

    def filter(self, record):
        """Set each :class:`~logging.LogRecord`'s `username` field."""
        record.username = self.username
        return 1