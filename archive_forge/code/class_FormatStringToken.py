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
class FormatStringToken(collections.namedtuple('FormatStringToken', 'text, name')):
    """
    A named tuple for the results of :func:`FormatStringParser.get_pairs()`.

    .. attribute:: name

       The field name referenced in `text` (a string). If `text` doesn't
       contain a formatting directive this will be :data:`None`.

    .. attribute:: text

       The text extracted from the logging format string (a string).
    """