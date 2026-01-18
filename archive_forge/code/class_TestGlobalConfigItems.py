import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
class TestGlobalConfigItems(tests.TestCaseInTempDir):

    def _get_empty_config(self):
        my_config = config.GlobalConfig()
        return my_config

    def _get_sample_config(self):
        my_config = config.GlobalConfig.from_string(sample_config_text)
        return my_config

    def test_user_id(self):
        my_config = config.GlobalConfig.from_string(sample_config_text)
        self.assertEqual('Erik Bågfors <erik@bagfors.nu>', my_config._get_user_id())

    def test_absent_user_id(self):
        my_config = config.GlobalConfig()
        self.assertEqual(None, my_config._get_user_id())

    def test_get_user_option_default(self):
        my_config = self._get_empty_config()
        self.assertEqual(None, my_config.get_user_option('no_option'))

    def test_get_user_option_global(self):
        my_config = self._get_sample_config()
        self.assertEqual('something', my_config.get_user_option('user_global_option'))

    def test_configured_validate_signatures_in_log(self):
        my_config = self._get_sample_config()
        self.assertEqual(True, my_config.validate_signatures_in_log())

    def test_get_alias(self):
        my_config = self._get_sample_config()
        self.assertEqual('help', my_config.get_alias('h'))

    def test_get_aliases(self):
        my_config = self._get_sample_config()
        aliases = my_config.get_aliases()
        self.assertEqual(2, len(aliases))
        sorted_keys = sorted(aliases)
        self.assertEqual('help', aliases[sorted_keys[0]])
        self.assertEqual(sample_long_alias, aliases[sorted_keys[1]])

    def test_get_no_alias(self):
        my_config = self._get_sample_config()
        self.assertEqual(None, my_config.get_alias('foo'))

    def test_get_long_alias(self):
        my_config = self._get_sample_config()
        self.assertEqual(sample_long_alias, my_config.get_alias('ll'))

    def test_get_change_editor(self):
        my_config = self._get_sample_config()
        change_editor = my_config.get_change_editor('old', 'new')
        self.assertIs(diff.DiffFromTool, change_editor.__class__)
        self.assertEqual('vimdiff -of {new_path} {old_path}', ' '.join(change_editor.command_template))

    def test_get_no_change_editor(self):
        my_config = self._get_empty_config()
        change_editor = my_config.get_change_editor('old', 'new')
        self.assertIs(None, change_editor)

    def test_get_merge_tools(self):
        conf = self._get_sample_config()
        tools = conf.get_merge_tools()
        self.log(repr(tools))
        self.assertEqual({'funkytool': 'funkytool "arg with spaces" {this_temp}', 'sometool': 'sometool {base} {this} {other} -o {result}', 'newtool': '"newtool with spaces" {this_temp}'}, tools)

    def test_get_merge_tools_empty(self):
        conf = self._get_empty_config()
        tools = conf.get_merge_tools()
        self.assertEqual({}, tools)

    def test_find_merge_tool(self):
        conf = self._get_sample_config()
        cmdline = conf.find_merge_tool('sometool')
        self.assertEqual('sometool {base} {this} {other} -o {result}', cmdline)

    def test_find_merge_tool_not_found(self):
        conf = self._get_sample_config()
        cmdline = conf.find_merge_tool('DOES NOT EXIST')
        self.assertIs(cmdline, None)

    def test_find_merge_tool_known(self):
        conf = self._get_empty_config()
        cmdline = conf.find_merge_tool('kdiff3')
        self.assertEqual('kdiff3 {base} {this} {other} -o {result}', cmdline)

    def test_find_merge_tool_override_known(self):
        conf = self._get_empty_config()
        conf.set_user_option('bzr.mergetool.kdiff3', 'kdiff3 blah')
        cmdline = conf.find_merge_tool('kdiff3')
        self.assertEqual('kdiff3 blah', cmdline)