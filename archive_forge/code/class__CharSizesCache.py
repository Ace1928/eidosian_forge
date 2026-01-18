from __future__ import unicode_literals
import inspect
import os
import signal
import sys
import threading
import weakref
from wcwidth import wcwidth
from six.moves import range
class _CharSizesCache(dict):
    """
    Cache for wcwidth sizes.
    """

    def __missing__(self, string):
        if len(string) == 1:
            result = max(0, wcwidth(string))
        else:
            result = sum((max(0, wcwidth(c)) for c in string))
        if len(string) < 256:
            self[string] = result
        return result