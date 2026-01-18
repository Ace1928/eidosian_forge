from __future__ import annotations
import compileall
import itertools
import sys
import zipfile
from importlib.abc import PathEntryFinder
from types import ModuleType
from typing import Any, Generator
from typing_extensions import Protocol
import twisted
from twisted.python import modules
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.python.test.test_zippath import zipit
from twisted.trial.unittest import TestCase
class PathModificationTests(TwistedModulesTestCase):
    """
    These tests share setup/cleanup behavior of creating a dummy package and
    stuffing some code in it.
    """
    _serialnum = itertools.count()

    def setUp(self) -> None:
        self.pathExtensionName = self.mktemp()
        self.pathExtension = FilePath(self.pathExtensionName)
        self.pathExtension.createDirectory()
        self.packageName = 'pyspacetests%d' % (next(self._serialnum),)
        self.packagePath = self.pathExtension.child(self.packageName)
        self.packagePath.createDirectory()
        self.packagePath.child('__init__.py').setContent(b'')
        self.packagePath.child('a.py').setContent(b'')
        self.packagePath.child('b.py').setContent(b'')
        self.packagePath.child('c__init__.py').setContent(b'')
        self.pathSetUp = False

    def _setupSysPath(self) -> None:
        assert not self.pathSetUp
        self.pathSetUp = True
        sys.path.append(self.pathExtensionName)

    def _underUnderPathTest(self, doImport: bool=True) -> None:
        moddir2 = self.mktemp()
        fpmd = FilePath(moddir2)
        fpmd.createDirectory()
        fpmd.child('foozle.py').setContent(b'x = 123\n')
        self.packagePath.child('__init__.py').setContent(networkString(f'__path__.append({repr(moddir2)})\n'))
        self._setupSysPath()
        modinfo = modules.getModule(self.packageName)
        self.assertEqual(self.findByIteration(self.packageName + '.foozle', modinfo, importPackages=doImport), modinfo['foozle'])
        self.assertEqual(modinfo['foozle'].load().x, 123)

    def test_underUnderPathAlreadyImported(self) -> None:
        """
        Verify that iterModules will honor the __path__ of already-loaded packages.
        """
        self._underUnderPathTest()

    def _listModules(self) -> None:
        pkginfo = modules.getModule(self.packageName)
        nfni = [modinfo.name.split('.')[-1] for modinfo in pkginfo.iterModules()]
        nfni.sort()
        self.assertEqual(nfni, ['a', 'b', 'c__init__'])

    def test_listingModules(self) -> None:
        """
        Make sure the module list comes back as we expect from iterModules on a
        package, whether zipped or not.
        """
        self._setupSysPath()
        self._listModules()

    def test_listingModulesAlreadyImported(self) -> None:
        """
        Make sure the module list comes back as we expect from iterModules on a
        package, whether zipped or not, even if the package has already been
        imported.
        """
        self._setupSysPath()
        namedAny(self.packageName)
        self._listModules()

    def tearDown(self) -> None:
        if self.pathSetUp:
            HORK = "path cleanup failed: don't be surprised if other tests break"
            assert sys.path.pop() is self.pathExtensionName, HORK + ', 1'
            assert self.pathExtensionName not in sys.path, HORK + ', 2'