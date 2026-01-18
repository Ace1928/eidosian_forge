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
def _win32_rename(old, new):
    """We expect to be able to atomically replace 'new' with old.

    On win32, if new exists, it must be moved out of the way first,
    and then deleted.
    """
    try:
        fancy_rename(old, new, rename_func=os.rename, unlink_func=os.unlink)
    except OSError as e:
        if e.errno in (errno.EPERM, errno.EACCES, errno.EBUSY, errno.EINVAL):
            os.lstat(old)
        raise