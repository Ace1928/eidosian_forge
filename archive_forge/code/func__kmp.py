import collections
import datetime
import functools
import os
import subprocess
import sys
import time
import errno
from contextlib import contextmanager
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import __version__ as PIL__version__
from PIL import ImageGrab
def _kmp(needle, haystack, _dummy):
    """
    TODO
    """
    shifts = [1] * (len(needle) + 1)
    shift = 1
    for pos in range(len(needle)):
        while shift <= pos and needle[pos] != needle[pos - shift]:
            shift += shifts[pos - shift]
        shifts[pos + 1] = shift
    startPos = 0
    matchLen = 0
    for c in haystack:
        while matchLen == len(needle) or (matchLen >= 0 and needle[matchLen] != c):
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(needle):
            yield startPos