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
def _logScreenshot(logScreenshot, funcName, funcArgs, folder='.'):
    """
    A helper function that creates a screenshot to act as a logging mechanism. When a PyAutoGUI function is called,
    this function is also called to capture the state of the screen when that function was called.

    If ``logScreenshot`` is ``False`` (or None and the ``LOG_SCREENSHOTS`` constant is ``False``), no screenshot is taken.

    The ``funcName`` argument is a string of the calling function's name. It's used in the screenshot's filename.

    The ``funcArgs`` argument is a string describing the arguments passed to the calling function. It's limited to
    twelve characters to keep it short.

    The ``folder`` argument is the folder to place the screenshot file in, and defaults to the current working directory.
    """
    if not logScreenshot:
        return
    if logScreenshot is None and LOG_SCREENSHOTS is False:
        return
    if len(funcArgs) > 12:
        funcArgs = funcArgs[:12] + '...'
    now = datetime.datetime.now()
    filename = '%s-%s-%s_%s-%s-%s-%s_%s_%s.png' % (now.year, str(now.month).rjust(2, '0'), str(now.day).rjust(2, '0'), now.hour, now.minute, now.second, str(now.microsecond)[:3], funcName, funcArgs)
    filepath = os.path.join(folder, filename)
    if LOG_SCREENSHOTS_LIMIT is not None and len(G_LOG_SCREENSHOTS_FILENAMES) >= LOG_SCREENSHOTS_LIMIT:
        os.unlink(os.path.join(folder, G_LOG_SCREENSHOTS_FILENAMES[0]))
        del G_LOG_SCREENSHOTS_FILENAMES[0]
    screenshot(filepath)
    G_LOG_SCREENSHOTS_FILENAMES.append(filename)