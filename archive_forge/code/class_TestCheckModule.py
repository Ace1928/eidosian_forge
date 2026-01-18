from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
class TestCheckModule(unittest.TestCase):

    def setUp(self):
        self.dir = TempDir(prefix='test_dir_module', dir='.')
        self.dir.__enter__()
        sys.path.insert(0, os.getcwd())

    def tearDown(self):
        sys.path.pop(0)
        self.dir.__exit__()

    def test_module(self):
        self._mkfile('module.py')
        path = _check_module.find('module')
        self.assertEqual(path, os.path.abspath('module.py'))
        self.assertNotIn('module', sys.modules)

    def test_package(self):
        os.mkdir('package')
        self._mkfile('package/__init__.py')
        self._mkfile('package/module.py')
        path = _check_module.find('package.module')
        self.assertEqual(path, os.path.abspath('package/module.py'))
        self.assertNotIn('package', sys.modules)
        self.assertNotIn('package.module', sys.modules)

    def test_subpackage(self):
        os.mkdir('package')
        self._mkfile('package/__init__.py')
        os.mkdir('package/subpackage')
        self._mkfile('package/subpackage/__init__.py')
        self._mkfile('package/subpackage/module.py')
        path = _check_module.find('package.subpackage.module')
        self.assertEqual(path, os.path.abspath('package/subpackage/module.py'))
        self.assertNotIn('package', sys.modules)
        self.assertNotIn('package.subpackage', sys.modules)
        self.assertNotIn('package.subpackage.module', sys.modules)

    def test_package_main(self):
        os.mkdir('package')
        self._mkfile('package/__init__.py')
        self._mkfile('package/__main__.py')
        path = _check_module.find('package')
        self.assertEqual(path, os.path.abspath('package/__main__.py'))
        self.assertNotIn('package', sys.modules)

    def test_not_package(self):
        self._mkfile('module.py')
        with self.assertRaisesRegexp(Exception, 'module is not a package'):
            _check_module.find('module.bad')
        self.assertNotIn('module', sys.modules)

    def _mkfile(self, path):
        open(path, 'w').close()