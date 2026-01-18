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
class StdinIteratorCls(six.Iterator):
    """An iterator that returns lines from stdin.
     This is needed because Python 3 balks at pickling the
     generator version above.
  """

    def __iter__(self):
        return self

    def __next__(self):
        line = sys.stdin.readline()
        if not line:
            raise StopIteration()
        return line.rstrip()