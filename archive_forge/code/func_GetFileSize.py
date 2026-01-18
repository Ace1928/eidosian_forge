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
def GetFileSize(fp, position_to_eof=False):
    """Returns size of file, optionally leaving fp positioned at EOF."""
    if not position_to_eof:
        cur_pos = fp.tell()
    fp.seek(0, os.SEEK_END)
    cur_file_size = fp.tell()
    if not position_to_eof:
        fp.seek(cur_pos)
    return cur_file_size