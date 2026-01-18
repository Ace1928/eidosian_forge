import sys
import functools
import difflib
import pprint
import re
import warnings
import collections
import contextlib
import traceback
import types
from . import result
from .util import (strclass, safe_repr, _count_diff_all_purpose,
def _formatMessage(self, msg, standardMsg):
    """Honour the longMessage attribute when generating failure messages.
        If longMessage is False this means:
        * Use only an explicit message if it is provided
        * Otherwise use the standard message for the assert

        If longMessage is True:
        * Use the standard message
        * If an explicit message is provided, plus ' : ' and the explicit message
        """
    if not self.longMessage:
        return msg or standardMsg
    if msg is None:
        return standardMsg
    try:
        return '%s : %s' % (standardMsg, msg)
    except UnicodeDecodeError:
        return '%s : %s' % (safe_repr(standardMsg), safe_repr(msg))