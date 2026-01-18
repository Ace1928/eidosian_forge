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
def GetStreamFromFileUrl(storage_url, mode='rb'):
    if storage_url.IsStream():
        return sys.stdin if six.PY2 else sys.stdin.buffer
    else:
        return open(storage_url.object_name, mode)