from __future__ import print_function, unicode_literals
import typing
import array
import calendar
import datetime
import io
import itertools
import socket
import threading
from collections import OrderedDict
from contextlib import contextmanager
from ftplib import FTP
from typing import cast
from ftplib import error_perm, error_temp
from six import PY2, raise_from, text_type
from . import _ftp_parse as ftp_parse
from . import errors
from .base import FS
from .constants import DEFAULT_CHUNK_SIZE
from .enums import ResourceType, Seek
from .info import Info
from .iotools import line_iterator
from .mode import Mode
from .path import abspath, basename, dirname, normpath, split
from .time import epoch_to_datetime
@classmethod
def _parse_ftp_time(cls, time_text):
    """Parse a time from an ftp directory listing."""
    try:
        tm_year = int(time_text[0:4])
        tm_month = int(time_text[4:6])
        tm_day = int(time_text[6:8])
        tm_hour = int(time_text[8:10])
        tm_min = int(time_text[10:12])
        tm_sec = int(time_text[12:14])
    except ValueError:
        return None
    epoch_time = calendar.timegm((tm_year, tm_month, tm_day, tm_hour, tm_min, tm_sec))
    return epoch_time