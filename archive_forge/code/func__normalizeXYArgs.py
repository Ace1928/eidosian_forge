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
def _normalizeXYArgs(firstArg, secondArg):
    """
    Returns a ``Point`` object based on ``firstArg`` and ``secondArg``, which are the first two arguments passed to
    several PyAutoGUI functions. If ``firstArg`` and ``secondArg`` are both ``None``, returns the current mouse cursor
    position.

    ``firstArg`` and ``secondArg`` can be integers, a sequence of integers, or a string representing an image filename
    to find on the screen (and return the center coordinates of).
    """
    if firstArg is None and secondArg is None:
        return position()
    elif firstArg is None and secondArg is not None:
        return Point(int(position()[0]), int(secondArg))
    elif secondArg is None and firstArg is not None and (not isinstance(firstArg, Sequence)):
        return Point(int(firstArg), int(position()[1]))
    elif isinstance(firstArg, str):
        try:
            location = locateOnScreen(firstArg)
            if location is not None:
                return center(location)
            else:
                return None
        except pyscreeze.ImageNotFoundException:
            raise ImageNotFoundException
        return center(locateOnScreen(firstArg))
    elif isinstance(firstArg, Sequence):
        if len(firstArg) == 2:
            if secondArg is None:
                return Point(int(firstArg[0]), int(firstArg[1]))
            else:
                raise PyAutoGUIException('When passing a sequence for firstArg, secondArg must not be passed (received {0}).'.format(repr(secondArg)))
        elif len(firstArg) == 4:
            if secondArg is None:
                return center(firstArg)
            else:
                raise PyAutoGUIException('When passing a sequence for firstArg, secondArg must not be passed and default to None (received {0}).'.format(repr(secondArg)))
        else:
            raise PyAutoGUIException('The supplied sequence must have exactly 2 or exactly 4 elements ({0} were received).'.format(len(firstArg)))
    else:
        return Point(int(firstArg), int(secondArg))