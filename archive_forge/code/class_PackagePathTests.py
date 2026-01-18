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
class PackagePathTests(unittest.TestCase):
    """
    Tests for L{plugin.pluginPackagePaths} which constructs search paths for
    plugin packages.
    """

    def setUp(self) -> None:
        """
        Save the elements of C{sys.path}.
        """
        self.originalPath = sys.path[:]

    def tearDown(self) -> None:
        """
        Restore C{sys.path} to its original value.
        """
        sys.path[:] = self.originalPath

    def test_pluginDirectories(self) -> None:
        """
        L{plugin.pluginPackagePaths} should return a list containing each
        directory in C{sys.path} with a suffix based on the supplied package
        name.
        """
        foo = FilePath('foo')
        bar = FilePath('bar')
        sys.path = [foo.path, bar.path]
        self.assertEqual(plugin.pluginPackagePaths('dummy.plugins'), [foo.child('dummy').child('plugins').path, bar.child('dummy').child('plugins').path])

    def test_pluginPackagesExcluded(self) -> None:
        """
        L{plugin.pluginPackagePaths} should exclude directories which are
        Python packages.  The only allowed plugin package (the only one
        associated with a I{dummy} package which Python will allow to be
        imported) will already be known to the caller of
        L{plugin.pluginPackagePaths} and will most commonly already be in
        the C{__path__} they are about to mutate.
        """
        root = FilePath(self.mktemp())
        foo = root.child('foo').child('dummy').child('plugins')
        foo.makedirs()
        foo.child('__init__.py').setContent(b'')
        sys.path = [root.child('foo').path, root.child('bar').path]
        self.assertEqual(plugin.pluginPackagePaths('dummy.plugins'), [root.child('bar').child('dummy').child('plugins').path])