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
@functools.lru_cache(maxsize=None)
def _macos_vers():
    version = platform.mac_ver()[0]
    if version == '':
        plist = '/System/Library/CoreServices/SystemVersion.plist'
        if os.path.exists(plist):
            with open(plist, 'rb') as fh:
                plist_content = plistlib.load(fh)
            if 'ProductVersion' in plist_content:
                version = plist_content['ProductVersion']
    return version.split('.')