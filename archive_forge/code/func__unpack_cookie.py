import os
import abc
import codecs
import errno
import stat
import sys
from _thread import allocate_lock as Lock
import io
from io import (__all__, SEEK_SET, SEEK_CUR, SEEK_END)
from _io import FileIO
def _unpack_cookie(self, bigint):
    rest, position = divmod(bigint, 1 << 64)
    rest, dec_flags = divmod(rest, 1 << 64)
    rest, bytes_to_feed = divmod(rest, 1 << 64)
    need_eof, chars_to_skip = divmod(rest, 1 << 64)
    return (position, dec_flags, bytes_to_feed, bool(need_eof), chars_to_skip)