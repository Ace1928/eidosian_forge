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
def _posix_is_local_pid_dead(pid):
    """True if pid doesn't correspond to live process on this machine"""
    try:
        os.kill(pid, 0)
    except OSError as e:
        if e.errno == errno.ESRCH:
            return True
        elif e.errno == errno.EPERM:
            return False
        else:
            trace.mutter('os.kill(%d, 0) failed: %s' % (pid, e))
            return False
    else:
        return False