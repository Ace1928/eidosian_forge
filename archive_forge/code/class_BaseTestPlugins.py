import importlib
import logging
import os
import sys
import types
from io import StringIO
from typing import Any, Dict, List
import breezy
from .. import osutils, plugin, tests
from . import test_bar
class BaseTestPlugins(tests.TestCaseInTempDir):
    """TestCase that isolates plugin imports and cleans up on completion."""

    def setUp(self):
        super().setUp()
        self.module_name = 'breezy.testingplugins'
        self.module_prefix = self.module_name + '.'
        self.module = types.ModuleType(self.module_name)
        self.overrideAttr(plugin, '_MODULE_PREFIX', self.module_prefix)
        self.overrideAttr(breezy, 'testingplugins', self.module)
        sys.modules[self.module_name] = self.module
        self.addCleanup(self._unregister_all)
        self.addCleanup(self._unregister_finder)
        invalidate_caches()

    def reset(self):
        """Remove all global testing state and clean up module."""
        self.log('resetting plugin testing context')
        self._unregister_all()
        self._unregister_finder()
        sys.modules[self.module_name] = self.module
        for name in list(self.module.__dict__):
            if name[:2] != '__':
                delattr(self.module, name)
        invalidate_caches()
        self.plugins = None

    def update_module_paths(self, paths):
        paths = plugin.extend_path(paths, self.module_name)
        self.module.__path__ = paths
        self.log('using %r', paths)
        return paths

    def load_with_paths(self, paths, warn_load_problems=True):
        self.log('loading plugins!')
        plugin.load_plugins(self.update_module_paths(paths), state=self, warn_load_problems=warn_load_problems)

    def create_plugin(self, name, source=None, dir='.', file_name=None):
        if source is None:
            source = '"""This is the doc for %s"""\n' % name
        if file_name is None:
            file_name = name + '.py'
        path = osutils.pathjoin(dir, file_name)
        with open(path, 'w') as f:
            f.write(source + '\n')

    def create_plugin_package(self, name, dir=None, source=None):
        if dir is None:
            dir = name
        if source is None:
            source = '"""This is the doc for {}"""\ndir_source = \'{}\'\n'.format(name, dir)
        os.makedirs(dir)
        self.create_plugin(name, source, dir, file_name='__init__.py')

    def promote_cache(self, directory):
        """Move bytecode files out of __pycache__ in given directory."""
        cache_dir = os.path.join(directory, '__pycache__')
        if os.path.isdir(cache_dir):
            for name in os.listdir(cache_dir):
                magicless_name = '.'.join(name.split('.')[0::name.count('.')])
                rel = osutils.relpath(self.test_dir, cache_dir)
                self.log('moving %s in %s to %s', name, rel, magicless_name)
                os.rename(os.path.join(cache_dir, name), os.path.join(directory, magicless_name))

    def _unregister_finder(self):
        """Removes any test copies of _PluginsAtFinder from sys.meta_path."""
        idx = len(sys.meta_path)
        while idx:
            idx -= 1
            finder = sys.meta_path[idx]
            if getattr(finder, 'prefix', '') == self.module_prefix:
                self.log('removed %r from sys.meta_path', finder)
                sys.meta_path.pop(idx)

    def _unregister_all(self):
        """Remove all plugins in the test namespace from sys.modules."""
        for name in list(sys.modules):
            if name.startswith(self.module_prefix) or name == self.module_name:
                self.log('removed %s from sys.modules', name)
                del sys.modules[name]

    def assertPluginModules(self, plugin_dict):
        self.assertEqual({k[len(self.module_prefix):]: sys.modules[k] for k in sys.modules if k.startswith(self.module_prefix)}, plugin_dict)

    def assertPluginUnknown(self, name):
        self.assertTrue(getattr(self.module, name, None) is None)
        self.assertFalse(self.module_prefix + name in sys.modules)

    def assertPluginKnown(self, name):
        self.assertTrue(getattr(self.module, name, None) is not None, 'plugins known: %r' % dir(self.module))
        self.assertTrue(self.module_prefix + name in sys.modules)