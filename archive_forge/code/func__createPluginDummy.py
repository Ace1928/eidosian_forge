from __future__ import annotations
import compileall
import errno
import functools
import os
import sys
import time
from importlib import invalidate_caches as invalidateImportCaches
from types import ModuleType
from typing import Callable, TypedDict, TypeVar
from zope.interface import Interface
from twisted import plugin
from twisted.python.filepath import FilePath
from twisted.python.log import EventDict, addObserver, removeObserver, textFromEventDict
from twisted.trial import unittest
from twisted.plugin import pluginPackagePaths
def _createPluginDummy(entrypath: FilePath[str], pluginContent: bytes, real: bool, pluginModule: str) -> FilePath[str]:
    """
    Create a plugindummy package.
    """
    entrypath.createDirectory()
    pkg = entrypath.child('plugindummy')
    pkg.createDirectory()
    if real:
        pkg.child('__init__.py').setContent(b'')
    plugs = pkg.child('plugins')
    plugs.createDirectory()
    if real:
        plugs.child('__init__.py').setContent(pluginInitFile)
    plugs.child(pluginModule + '.py').setContent(pluginContent)
    return plugs