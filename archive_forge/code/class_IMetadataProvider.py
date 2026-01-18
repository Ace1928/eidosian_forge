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
class IMetadataProvider(Protocol):

    def has_metadata(self, name):
        """Does the package's distribution contain the named metadata?"""

    def get_metadata(self, name):
        """The named metadata resource as a string"""

    def get_metadata_lines(self, name):
        """Yield named metadata resource as list of non-blank non-comment lines

        Leading and trailing whitespace is stripped from each line, and lines
        with ``#`` as the first non-blank character are omitted."""

    def metadata_isdir(self, name):
        """Is the named metadata a directory?  (like ``os.path.isdir()``)"""

    def metadata_listdir(self, name):
        """List of metadata names in the directory (like ``os.listdir()``)"""

    def run_script(self, script_name, namespace):
        """Execute the named script in the supplied namespace dictionary"""