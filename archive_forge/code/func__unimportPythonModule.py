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
def _unimportPythonModule(self, module: ModuleType, deleteSource: bool=False) -> None:
    assert module.__file__ is not None
    modulePath = module.__name__.split('.')
    packageName = '.'.join(modulePath[:-1])
    moduleName = modulePath[-1]
    delattr(sys.modules[packageName], moduleName)
    del sys.modules[module.__name__]
    for ext in ['c', 'o'] + (deleteSource and [''] or []):
        try:
            os.remove(module.__file__ + ext)
        except FileNotFoundError:
            pass