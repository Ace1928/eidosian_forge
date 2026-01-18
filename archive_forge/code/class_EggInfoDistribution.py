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
class EggInfoDistribution(Distribution):

    def _reload_version(self):
        """
        Packages installed by distutils (e.g. numpy or scipy),
        which uses an old safe_version, and so
        their version numbers can get mangled when
        converted to filenames (e.g., 1.11.0.dev0+2329eae to
        1.11.0.dev0_2329eae). These distributions will not be
        parsed properly
        downstream by Distribution and safe_version, so
        take an extra step and try to get the version number from
        the metadata file itself instead of the filename.
        """
        md_version = self._get_version()
        if md_version:
            self._version = md_version
        return self