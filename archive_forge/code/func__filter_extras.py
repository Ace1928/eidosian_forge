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
@staticmethod
def _filter_extras(dm):
    """
        Given a mapping of extras to dependencies, strip off
        environment markers and filter out any dependencies
        not matching the markers.
        """
    for extra in list(filter(None, dm)):
        new_extra = extra
        reqs = dm.pop(extra)
        new_extra, _, marker = extra.partition(':')
        fails_marker = marker and (invalid_marker(marker) or not evaluate_marker(marker))
        if fails_marker:
            reqs = []
        new_extra = safe_extra(new_extra) or None
        dm.setdefault(new_extra, []).extend(reqs)
    return dm