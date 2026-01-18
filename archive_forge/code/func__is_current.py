import sys
import os
import io
import time
import re
import types
from typing import Protocol
import zipfile
import zipimport
import warnings
import stat
import functools
import pkgutil
import operator
import platform
import collections
import plistlib
import email.parser
import errno
import tempfile
import textwrap
import inspect
import ntpath
import posixpath
import importlib
import importlib.machinery
from pkgutil import get_importer
import _imp
from os import utime
from os import open as os_open
from os.path import isdir, split
from pkg_resources.extern.jaraco.text import (
from pkg_resources.extern import platformdirs
from pkg_resources.extern import packaging
def _is_current(self, file_path, zip_path):
    """
        Return True if the file_path is current for this zip_path
        """
    timestamp, size = self._get_date_and_size(self.zipinfo[zip_path])
    if not os.path.isfile(file_path):
        return False
    stat = os.stat(file_path)
    if stat.st_size != size or stat.st_mtime != timestamp:
        return False
    zip_contents = self.loader.get_data(zip_path)
    with open(file_path, 'rb') as f:
        file_contents = f.read()
    return zip_contents == file_contents