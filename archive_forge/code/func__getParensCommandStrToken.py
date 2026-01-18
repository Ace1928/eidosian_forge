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
def _getParensCommandStrToken(commandStr):
    """Gets the command string token at the start of commandStr. It will also
    be enclosed with parentheses.

    Given "(ccc)world" returns "(ccc)"
    Given "  (ccc)world" returns "  (ccc)"
    Given "(ccf10(r))world" returns "(ccf10(r))"

    Raises an exception if it can't tokenize a quoted string.
    """
    pattern = re.compile('^\\s*\\(')
    mo = pattern.search(commandStr)
    if mo is None:
        raise PyAutoGUIException('Invalid command at index 0: No open parenthesis found.')
    i = 0
    openParensCount = 0
    while i < len(commandStr):
        if commandStr[i] == '(':
            openParensCount += 1
        elif commandStr[i] == ')':
            openParensCount -= 1
            if openParensCount == 0:
                i += 1
                break
            elif openParensCount == -1:
                raise PyAutoGUIException('Invalid command at index 0: No open parenthesis for this close parenthesis.')
        i += 1
    if openParensCount > 0:
        raise PyAutoGUIException('Invalid command at index 0: Not enough close parentheses.')
    return commandStr[0:i]