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
class DeveloperSetupTests(unittest.TestCase):
    """
    These tests verify things about the plugin system without actually
    interacting with the deployed 'twisted.plugins' package, instead creating a
    temporary package.
    """

    def setUp(self) -> None:
        """
        Create a complex environment with multiple entries on sys.path, akin to
        a developer's environment who has a development (trunk) checkout of
        Twisted, a system installed version of Twisted (for their operating
        system's tools) and a project which provides Twisted plugins.
        """
        self.savedPath = sys.path[:]
        self.savedModules = sys.modules.copy()
        self.fakeRoot = FilePath(self.mktemp())
        self.fakeRoot.createDirectory()
        self.systemPath = self.fakeRoot.child('system_path')
        self.devPath = self.fakeRoot.child('development_path')
        self.appPath = self.fakeRoot.child('application_path')
        self.systemPackage = _createPluginDummy(self.systemPath, pluginFileContents('system'), True, 'plugindummy_builtin')
        self.devPackage = _createPluginDummy(self.devPath, pluginFileContents('dev'), True, 'plugindummy_builtin')
        self.appPackage = _createPluginDummy(self.appPath, pluginFileContents('app'), False, 'plugindummy_app')
        sys.path.extend([x.path for x in [self.systemPath, self.appPath]])
        self.getAllPlugins()
        self.sysplug = self.systemPath.child('plugindummy').child('plugins')
        self.syscache = self.sysplug.child('dropin.cache')
        now = time.time()
        os.utime(self.sysplug.child('plugindummy_builtin.py').path, (now - 5000,) * 2)
        os.utime(self.syscache.path, (now - 2000,) * 2)
        self.lockSystem()
        self.resetEnvironment()

    def lockSystem(self) -> None:
        """
        Lock the system directories, as if they were unwritable by this user.
        """
        os.chmod(self.sysplug.path, 365)
        os.chmod(self.syscache.path, 365)

    def unlockSystem(self) -> None:
        """
        Unlock the system directories, as if they were writable by this user.
        """
        os.chmod(self.sysplug.path, 511)
        os.chmod(self.syscache.path, 511)

    def getAllPlugins(self) -> list[str]:
        """
        Get all the plugins loadable from our dummy package, and return their
        short names.
        """
        import plugindummy.plugins
        x = list(plugin.getPlugins(ITestPlugin, plugindummy.plugins))
        return [plug.__name__ for plug in x]

    def resetEnvironment(self) -> None:
        """
        Change the environment to what it should be just as the test is
        starting.
        """
        self.unsetEnvironment()
        sys.path.extend([x.path for x in [self.devPath, self.systemPath, self.appPath]])

    def unsetEnvironment(self) -> None:
        """
        Change the Python environment back to what it was before the test was
        started.
        """
        invalidateImportCaches()
        sys.modules.clear()
        sys.modules.update(self.savedModules)
        sys.path[:] = self.savedPath

    def tearDown(self) -> None:
        """
        Reset the Python environment to what it was before this test ran, and
        restore permissions on files which were marked read-only so that the
        directory may be cleanly cleaned up.
        """
        self.unsetEnvironment()
        self.unlockSystem()

    def test_developmentPluginAvailability(self) -> None:
        """
        Plugins added in the development path should be loadable, even when
        the (now non-importable) system path contains its own idea of the
        list of plugins for a package.  Inversely, plugins added in the
        system path should not be available.
        """
        for x in range(3):
            names = self.getAllPlugins()
            names.sort()
            self.assertEqual(names, ['app', 'dev'])

    def test_freshPyReplacesStalePyc(self) -> None:
        """
        Verify that if a stale .pyc file on the PYTHONPATH is replaced by a
        fresh .py file, the plugins in the new .py are picked up rather than
        the stale .pyc, even if the .pyc is still around.
        """
        mypath = self.appPackage.child('stale.py')
        mypath.setContent(pluginFileContents('one'))
        x = time.time() - 1000
        os.utime(mypath.path, (x, x))
        pyc = mypath.sibling('stale.pyc')
        extra = _HasBoolLegacyKey(legacy=True)
        compileall.compile_dir(self.appPackage.path, quiet=1, **extra)
        os.utime(pyc.path, (x, x))
        mypath.remove()
        self.resetEnvironment()
        self.assertIn('one', self.getAllPlugins())
        self.failIfIn('two', self.getAllPlugins())
        self.resetEnvironment()
        mypath.setContent(pluginFileContents('two'))
        self.failIfIn('one', self.getAllPlugins())
        self.assertIn('two', self.getAllPlugins())

    def test_newPluginsOnReadOnlyPath(self) -> None:
        """
        Verify that a failure to write the dropin.cache file on a read-only
        path will not affect the list of plugins returned.

        Note: this test should pass on both Linux and Windows, but may not
        provide useful coverage on Windows due to the different meaning of
        "read-only directory".
        """
        self.unlockSystem()
        self.sysplug.child('newstuff.py').setContent(pluginFileContents('one'))
        self.lockSystem()
        sys.path.remove(self.devPath.path)
        events: list[EventDict] = []
        addObserver(events.append)
        self.addCleanup(removeObserver, events.append)
        self.assertIn('one', self.getAllPlugins())
        expected = 'Unable to write to plugin cache %s: error number %d' % (self.syscache.path, errno.EPERM)
        for event in events:
            maybeText = textFromEventDict(event)
            assert maybeText is not None
            if expected in maybeText:
                break
        else:
            self.fail('Did not observe unwriteable cache warning in log events: %r' % (events,))