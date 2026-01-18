from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
def _getQuotedStringToken(commandStr):
    """Gets the quoted string token at the start of commandStr.
    The quoted string must use single quotes.

    Given "'hello'world" returns "'hello'"
    Given "  'hello'world" returns "  'hello'"

    Raises an exception if it can't tokenize a quoted string.
    """
    pattern = re.compile("^((\\s*)('(.*?)'))")
    mo = pattern.search(commandStr)
    if mo is None:
        raise PyAutoGUIException('Invalid command at index 0: a quoted string was expected')
    return mo.group(1)