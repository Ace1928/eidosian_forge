import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def _format_date(t, offset, timezone, date_fmt, show_offset):
    if timezone == 'utc':
        tt = time.gmtime(t)
        offset = 0
    elif timezone == 'original':
        if offset is None:
            offset = 0
        tt = time.gmtime(t + offset)
    elif timezone == 'local':
        tt = time.localtime(t)
        offset = local_time_offset(t)
    else:
        raise UnsupportedTimezoneFormat(timezone)
    if date_fmt is None:
        date_fmt = '%a %Y-%m-%d %H:%M:%S'
    if show_offset:
        offset_str = ' %+03d%02d' % (offset / 3600, offset / 60 % 60)
    else:
        offset_str = ''
    return (date_fmt, tt, offset_str)