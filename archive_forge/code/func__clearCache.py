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
def _clearCache(self) -> None:
    """
        Remove the plugins B{droping.cache} file.
        """
    self.package.child('dropin.cache').remove()