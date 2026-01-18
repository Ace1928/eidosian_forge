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
def _handle_ns(packageName, path_item):
    """Ensure that named package includes a subpath of path_item (if needed)"""
    importer = get_importer(path_item)
    if importer is None:
        return None
    try:
        spec = importer.find_spec(packageName)
    except AttributeError:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loader = importer.find_module(packageName)
    else:
        loader = spec.loader if spec else None
    if loader is None:
        return None
    module = sys.modules.get(packageName)
    if module is None:
        module = sys.modules[packageName] = types.ModuleType(packageName)
        module.__path__ = []
        _set_parent_ns(packageName)
    elif not hasattr(module, '__path__'):
        raise TypeError('Not a package:', packageName)
    handler = _find_adapter(_namespace_handlers, importer)
    subpath = handler(importer, path_item, packageName, module)
    if subpath is not None:
        path = module.__path__
        path.append(subpath)
        importlib.import_module(packageName)
        _rebuild_mod_path(path, packageName, module)
    return subpath