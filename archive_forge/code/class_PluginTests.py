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
class PluginTests(unittest.TestCase):
    """
    Tests which verify the behavior of the current, active Twisted plugins
    directory.
    """

    def setUp(self) -> None:
        """
        Save C{sys.path} and C{sys.modules}, and create a package for tests.
        """
        self.originalPath = sys.path[:]
        self.savedModules = sys.modules.copy()
        self.root = FilePath(self.mktemp())
        self.root.createDirectory()
        self.package = self.root.child('mypackage')
        self.package.createDirectory()
        self.package.child('__init__.py').setContent(b'')
        FilePath(__file__).sibling('plugin_basic.py').copyTo(self.package.child('testplugin.py'))
        self.originalPlugin = 'testplugin'
        sys.path.insert(0, self.root.path)
        import mypackage
        self.module = mypackage

    def tearDown(self) -> None:
        """
        Restore C{sys.path} and C{sys.modules} to their original values.
        """
        sys.path[:] = self.originalPath
        sys.modules.clear()
        sys.modules.update(self.savedModules)

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

    def _clearCache(self) -> None:
        """
        Remove the plugins B{droping.cache} file.
        """
        self.package.child('dropin.cache').remove()

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

    @_withCacheness
    def test_cache(self) -> None:
        """
        Check that the cache returned by L{plugin.getCache} hold the plugin
        B{testplugin}, and that this plugin has the properties we expect:
        provide L{TestPlugin}, has the good name and description, and can be
        loaded successfully.
        """
        cache = plugin.getCache(self.module)
        dropin = cache[self.originalPlugin]
        self.assertEqual(dropin.moduleName, f'mypackage.{self.originalPlugin}')
        self.assertIn("I'm a test drop-in.", dropin.description)
        p1 = [p for p in dropin.plugins if ITestPlugin in p.provided][0]
        self.assertIs(p1.dropin, dropin)
        self.assertEqual(p1.name, 'TestPlugin')
        self.assertEqual(p1.description.strip(), 'A plugin used solely for testing purposes.')
        self.assertEqual(p1.provided, [ITestPlugin, plugin.IPlugin])
        realPlugin = p1.load()
        self.assertIs(realPlugin, sys.modules[f'mypackage.{self.originalPlugin}'].TestPlugin)
        import mypackage.testplugin as tp
        self.assertIs(realPlugin, tp.TestPlugin)

    def test_cacheRepr(self) -> None:
        """
        L{CachedPlugin} has a helpful C{repr} which contains relevant
        information about it.
        """
        cachedDropin = plugin.getCache(self.module)[self.originalPlugin]
        cachedPlugin = list((p for p in cachedDropin.plugins if p.name == 'TestPlugin'))[0]
        self.assertEqual(repr(cachedPlugin), "<CachedPlugin 'TestPlugin'/'mypackage.testplugin' (provides 'ITestPlugin, IPlugin')>")

    @_withCacheness
    def test_plugins(self) -> None:
        """
        L{plugin.getPlugins} should return the list of plugins matching the
        specified interface (here, L{ITestPlugin2}), and these plugins
        should be instances of classes with a C{test} method, to be sure
        L{plugin.getPlugins} load classes correctly.
        """
        plugins = list(plugin.getPlugins(ITestPlugin2, self.module))
        self.assertEqual(len(plugins), 2)
        names = ['AnotherTestPlugin', 'ThirdTestPlugin']
        for p in plugins:
            names.remove(p.__name__)
            p.test()

    @_withCacheness
    def test_detectNewFiles(self) -> None:
        """
        Check that L{plugin.getPlugins} is able to detect plugins added at
        runtime.
        """
        FilePath(__file__).sibling('plugin_extra1.py').copyTo(self.package.child('pluginextra.py'))
        try:
            self.failIfIn('mypackage.pluginextra', sys.modules)
            self.assertFalse(hasattr(sys.modules['mypackage'], 'pluginextra'), 'mypackage still has pluginextra module')
            plgs = list(plugin.getPlugins(ITestPlugin, self.module))
            self.assertEqual(len(plgs), 2)
            names = ['TestPlugin', 'FourthTestPlugin']
            for p in plgs:
                names.remove(p.__name__)
                p.test1()
        finally:
            self._unimportPythonModule(sys.modules['mypackage.pluginextra'], True)

    @_withCacheness
    def test_detectFilesChanged(self) -> None:
        """
        Check that if the content of a plugin change, L{plugin.getPlugins} is
        able to detect the new plugins added.
        """
        FilePath(__file__).sibling('plugin_extra1.py').copyTo(self.package.child('pluginextra.py'))
        try:
            plgs = list(plugin.getPlugins(ITestPlugin, self.module))
            self.assertEqual(len(plgs), 2)
            FilePath(__file__).sibling('plugin_extra2.py').copyTo(self.package.child('pluginextra.py'))
            self._unimportPythonModule(sys.modules['mypackage.pluginextra'])
            plgs = list(plugin.getPlugins(ITestPlugin, self.module))
            self.assertEqual(len(plgs), 3)
            names = ['TestPlugin', 'FourthTestPlugin', 'FifthTestPlugin']
            for p in plgs:
                names.remove(p.__name__)
                p.test1()
        finally:
            self._unimportPythonModule(sys.modules['mypackage.pluginextra'], True)

    @_withCacheness
    def test_detectFilesRemoved(self) -> None:
        """
        Check that when a dropin file is removed, L{plugin.getPlugins} doesn't
        return it anymore.
        """
        FilePath(__file__).sibling('plugin_extra1.py').copyTo(self.package.child('pluginextra.py'))
        try:
            list(plugin.getPlugins(ITestPlugin, self.module))
        finally:
            self._unimportPythonModule(sys.modules['mypackage.pluginextra'], True)
        plgs = list(plugin.getPlugins(ITestPlugin, self.module))
        self.assertEqual(1, len(plgs))

    @_withCacheness
    def test_nonexistentPathEntry(self) -> None:
        """
        Test that getCache skips over any entries in a plugin package's
        C{__path__} which do not exist.
        """
        path = self.mktemp()
        self.assertFalse(os.path.exists(path))
        self.module.__path__.append(path)
        try:
            plgs = list(plugin.getPlugins(ITestPlugin, self.module))
            self.assertEqual(len(plgs), 1)
        finally:
            self.module.__path__.remove(path)

    @_withCacheness
    def test_nonDirectoryChildEntry(self) -> None:
        """
        Test that getCache skips over any entries in a plugin package's
        C{__path__} which refer to children of paths which are not directories.
        """
        path = FilePath(self.mktemp())
        self.assertFalse(path.exists())
        path.touch()
        child = path.child('test_package').path
        self.module.__path__.append(child)
        try:
            plgs = list(plugin.getPlugins(ITestPlugin, self.module))
            self.assertEqual(len(plgs), 1)
        finally:
            self.module.__path__.remove(child)

    def test_deployedMode(self) -> None:
        """
        The C{dropin.cache} file may not be writable: the cache should still be
        attainable, but an error should be logged to show that the cache
        couldn't be updated.
        """
        plugin.getCache(self.module)
        cachepath = self.package.child('dropin.cache')
        FilePath(__file__).sibling('plugin_extra1.py').copyTo(self.package.child('pluginextra.py'))
        invalidateImportCaches()
        os.chmod(self.package.path, 320)
        os.chmod(cachepath.path, 256)
        self.addCleanup(os.chmod, self.package.path, 448)
        self.addCleanup(os.chmod, cachepath.path, 448)
        events: list[EventDict] = []
        addObserver(events.append)
        self.addCleanup(removeObserver, events.append)
        cache = plugin.getCache(self.module)
        self.assertIn('pluginextra', cache)
        self.assertIn(self.originalPlugin, cache)
        expected = 'Unable to write to plugin cache %s: error number %d' % (cachepath.path, errno.EPERM)
        for event in events:
            maybeText = textFromEventDict(event)
            assert maybeText is not None
            if expected in maybeText:
                break
        else:
            self.fail('Did not observe unwriteable cache warning in log events: %r' % (events,))