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
class TestLoadingPlugins(BaseTestPlugins):
    activeattributes: Dict[str, List[Any]] = {}

    def test_plugins_with_the_same_name_are_not_loaded(self):
        tempattribute = '0'
        self.assertFalse(tempattribute in self.activeattributes)
        self.__class__.activeattributes[tempattribute] = []
        self.assertTrue(tempattribute in self.activeattributes)
        os.mkdir('first')
        os.mkdir('second')
        template = "from breezy.tests.test_plugins import TestLoadingPlugins\nTestLoadingPlugins.activeattributes[%r].append('%s')\n"
        with open(os.path.join('first', 'plugin.py'), 'w') as outfile:
            outfile.write(template % (tempattribute, 'first'))
            outfile.write('\n')
        with open(os.path.join('second', 'plugin.py'), 'w') as outfile:
            outfile.write(template % (tempattribute, 'second'))
            outfile.write('\n')
        try:
            self.load_with_paths(['first', 'second'])
            self.assertEqual(['first'], self.activeattributes[tempattribute])
        finally:
            del self.activeattributes[tempattribute]

    def test_plugins_from_different_dirs_can_demand_load(self):
        self.assertFalse('breezy.plugins.pluginone' in sys.modules)
        self.assertFalse('breezy.plugins.plugintwo' in sys.modules)
        tempattribute = 'different-dirs'
        self.assertFalse(tempattribute in self.activeattributes)
        breezy.tests.test_plugins.TestLoadingPlugins.activeattributes[tempattribute] = []
        self.assertTrue(tempattribute in self.activeattributes)
        os.mkdir('first')
        os.mkdir('second')
        template = "from breezy.tests.test_plugins import TestLoadingPlugins\nTestLoadingPlugins.activeattributes[%r].append('%s')\n"
        with open(os.path.join('first', 'pluginone.py'), 'w') as outfile:
            outfile.write(template % (tempattribute, 'first'))
            outfile.write('\n')
        with open(os.path.join('second', 'plugintwo.py'), 'w') as outfile:
            outfile.write(template % (tempattribute, 'second'))
            outfile.write('\n')
        try:
            self.assertPluginUnknown('pluginone')
            self.assertPluginUnknown('plugintwo')
            self.update_module_paths(['first', 'second'])
            exec('import %spluginone' % self.module_prefix)
            self.assertEqual(['first'], self.activeattributes[tempattribute])
            exec('import %splugintwo' % self.module_prefix)
            self.assertEqual(['first', 'second'], self.activeattributes[tempattribute])
        finally:
            del self.activeattributes[tempattribute]

    def test_plugins_can_load_from_directory_with_trailing_slash(self):
        self.assertPluginUnknown('ts_plugin')
        tempattribute = 'trailing-slash'
        self.assertFalse(tempattribute in self.activeattributes)
        breezy.tests.test_plugins.TestLoadingPlugins.activeattributes[tempattribute] = []
        self.assertTrue(tempattribute in self.activeattributes)
        os.mkdir('plugin_test')
        template = "from breezy.tests.test_plugins import TestLoadingPlugins\nTestLoadingPlugins.activeattributes[%r].append('%s')\n"
        with open(os.path.join('plugin_test', 'ts_plugin.py'), 'w') as outfile:
            outfile.write(template % (tempattribute, 'plugin'))
            outfile.write('\n')
        try:
            self.load_with_paths(['plugin_test' + os.sep])
            self.assertEqual(['plugin'], self.activeattributes[tempattribute])
            self.assertPluginKnown('ts_plugin')
        finally:
            del self.activeattributes[tempattribute]

    def load_and_capture(self, name, warn_load_problems=True):
        """Load plugins from '.' capturing the output.

        :param name: The name of the plugin.
        :return: A string with the log from the plugin loading call.
        """
        stream = StringIO()
        try:
            handler = logging.StreamHandler(stream)
            log = logging.getLogger('brz')
            log.addHandler(handler)
            try:
                self.load_with_paths(['.'], warn_load_problems=warn_load_problems)
            finally:
                handler.flush()
                handler.close()
                log.removeHandler(handler)
            return stream.getvalue()
        finally:
            stream.close()

    def test_plugin_with_bad_api_version_reports(self):
        """Try loading a plugin that requests an unsupported api.

        Observe that it records the problem but doesn't complain on stderr
        when warn_load_problems=False
        """
        name = 'wants100.py'
        with open(name, 'w') as f:
            f.write('import breezy\nfrom breezy.errors import IncompatibleVersion\nraise IncompatibleVersion(breezy, [(1, 0, 0)], (0, 0, 5))\n')
        log = self.load_and_capture(name, warn_load_problems=False)
        self.assertNotContainsRe(log, 'It supports breezy version')
        self.assertEqual({'wants100'}, self.plugin_warnings.keys())
        self.assertContainsRe(self.plugin_warnings['wants100'][0], 'It supports breezy version')

    def test_plugin_with_bad_name_does_not_load(self):
        name = 'brz-bad plugin-name..py'
        open(name, 'w').close()
        log = self.load_and_capture(name)
        self.assertContainsRe(log, "Unable to load 'brz-bad plugin-name\\.' in '.*' as a plugin because the file path isn't a valid module name; try renaming it to 'bad_plugin_name_'\\.")

    def test_plugin_with_error_suppress(self):
        name = 'some_error.py'
        with open(name, 'w') as f:
            f.write('raise Exception("bad")\n')
        log = self.load_and_capture(name, warn_load_problems=False)
        self.assertEqual('', log)

    def test_plugin_with_error(self):
        name = 'some_error.py'
        with open(name, 'w') as f:
            f.write('raise Exception("bad")\n')
        log = self.load_and_capture(name, warn_load_problems=True)
        self.assertContainsRe(log, "Unable to load plugin 'some_error' from '.*': bad\n")