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
class TestDisablePlugin(BaseTestPlugins):

    def test_cannot_import(self):
        self.create_plugin_package('works')
        self.create_plugin_package('fails')
        self.overrideEnv('BRZ_DISABLE_PLUGINS', 'fails')
        self.update_module_paths(['.'])
        import breezy.testingplugins.works as works
        try:
            import breezy.testingplugins.fails as fails
        except ImportError:
            pass
        else:
            self.fail('Loaded blocked plugin: ' + repr(fails))
        self.assertPluginModules({'fails': None, 'works': works})

    def test_partial_imports(self):
        self.create_plugin('good')
        self.create_plugin('bad')
        self.create_plugin_package('ugly')
        self.overrideEnv('BRZ_DISABLE_PLUGINS', 'bad:ugly')
        self.load_with_paths(['.'])
        self.assertEqual({'good'}, self.plugins.keys())
        self.assertPluginModules({'good': self.plugins['good'].module, 'bad': None, 'ugly': None})
        self.assertNotContainsRe(self.get_log(), 'Unable to load plugin')