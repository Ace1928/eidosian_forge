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
@property
def _forgiving_parsed_version(self):
    try:
        return self.parsed_version
    except packaging.version.InvalidVersion as ex:
        self._parsed_version = parse_version(_forgiving_version(self.version))
        notes = '\n'.join(getattr(ex, '__notes__', []))
        msg = f'!!\n\n\n            *************************************************************************\n            {str(ex)}\n{notes}\n\n            This is a long overdue deprecation.\n            For the time being, `pkg_resources` will use `{self._parsed_version}`\n            as a replacement to avoid breaking existing environments,\n            but no future compatibility is guaranteed.\n\n            If you maintain package {self.project_name} you should implement\n            the relevant changes to adequate the project to PEP 440 immediately.\n            *************************************************************************\n            \n\n!!\n            '
        warnings.warn(msg, DeprecationWarning)
        return self._parsed_version