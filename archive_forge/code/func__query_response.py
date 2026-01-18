import os
import re
import sys
import time
import codecs
import locale
import select
import struct
import platform
import warnings
import functools
import contextlib
import collections
from .color import COLOR_DISTANCE_ALGORITHMS
from .keyboard import (_time_left,
from .sequences import Termcap, Sequence, SequenceTextWrapper
from .colorspace import RGB_256TABLE
from .formatters import (COLORS,
from ._capabilities import CAPABILITY_DATABASE, CAPABILITIES_ADDITIVES, CAPABILITIES_RAW_MIXIN
def _query_response(self, query_str, response_re, timeout):
    """
        Sends a query string to the terminal and waits for a response.

        :arg str query_str: Query string written to output
        :arg str response_re: Regular expression matching query response
        :arg float timeout: Return after time elapsed in seconds
        :return: re.match object for response_re or None if not found
        :rtype: re.Match
        """
    ctx = None
    try:
        if self._line_buffered:
            ctx = self.cbreak()
            ctx.__enter__()
        self.stream.write(query_str)
        self.stream.flush()
        match, data = _read_until(term=self, pattern=response_re, timeout=timeout)
        if match:
            data = data[:match.start()] + data[match.end():]
        self.ungetch(data)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return match