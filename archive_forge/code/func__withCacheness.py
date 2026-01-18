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
def _withCacheness(meth: Callable[[PluginTests], object]) -> Callable[[PluginTests], None]:
    """
        This is a paranoid test wrapper, that calls C{meth} 2 times, clear the
        cache, and calls it 2 other times. It's supposed to ensure that the
        plugin system behaves correctly no matter what the state of the cache
        is.
        """

    @functools.wraps(meth)
    def wrapped(self: PluginTests) -> None:
        meth(self)
        meth(self)
        self._clearCache()
        meth(self)
        meth(self)
    return wrapped