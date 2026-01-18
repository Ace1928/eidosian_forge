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
class DefaultProvider(EggProvider):
    """Provides access to package resources in the filesystem"""

    def _has(self, path):
        return os.path.exists(path)

    def _isdir(self, path):
        return os.path.isdir(path)

    def _listdir(self, path):
        return os.listdir(path)

    def get_resource_stream(self, manager, resource_name):
        return open(self._fn(self.module_path, resource_name), 'rb')

    def _get(self, path):
        with open(path, 'rb') as stream:
            return stream.read()

    @classmethod
    def _register(cls):
        loader_names = ('SourceFileLoader', 'SourcelessFileLoader')
        for name in loader_names:
            loader_cls = getattr(importlib.machinery, name, type(None))
            register_loader_type(loader_cls, cls)