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
def _win32_delete_readonly(function, path, excinfo):
    """Error handler for shutil.rmtree function [for win32]
        Helps to remove files and dirs marked as read-only.
        """
    exception = excinfo[1]
    if function in (os.unlink, os.remove, os.rmdir) and isinstance(exception, OSError) and (exception.errno == errno.EACCES):
        make_writable(path)
        function(path)
    else:
        raise