from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import locale
import os
import struct
import sys
import six
from gslib.utils.constants import WINDOWS_1252
def StdinIterator():
    """A generator function that returns lines from stdin."""
    for line in sys.stdin:
        yield line.rstrip()