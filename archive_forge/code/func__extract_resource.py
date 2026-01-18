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
def _extract_resource(self, manager, zip_path):
    if zip_path in self._index():
        for name in self._index()[zip_path]:
            last = self._extract_resource(manager, os.path.join(zip_path, name))
        return os.path.dirname(last)
    timestamp, size = self._get_date_and_size(self.zipinfo[zip_path])
    if not WRITE_SUPPORT:
        raise OSError('"os.rename" and "os.unlink" are not supported on this platform')
    try:
        real_path = manager.get_cache_path(self.egg_name, self._parts(zip_path))
        if self._is_current(real_path, zip_path):
            return real_path
        outf, tmpnam = _mkstemp('.$extract', dir=os.path.dirname(real_path))
        os.write(outf, self.loader.get_data(zip_path))
        os.close(outf)
        utime(tmpnam, (timestamp, timestamp))
        manager.postprocess(tmpnam, real_path)
        try:
            rename(tmpnam, real_path)
        except OSError:
            if os.path.isfile(real_path):
                if self._is_current(real_path, zip_path):
                    return real_path
                elif os.name == 'nt':
                    unlink(real_path)
                    rename(tmpnam, real_path)
                    return real_path
            raise
    except OSError:
        manager.extraction_error()
    return real_path