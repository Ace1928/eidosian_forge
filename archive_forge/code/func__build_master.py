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
@classmethod
def _build_master(cls):
    """
        Prepare the master working set.
        """
    ws = cls()
    try:
        from __main__ import __requires__
    except ImportError:
        return ws
    try:
        ws.require(__requires__)
    except VersionConflict:
        return cls._build_from_requirements(__requires__)
    return ws