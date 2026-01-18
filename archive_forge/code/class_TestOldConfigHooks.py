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
class TestOldConfigHooks(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        create_configs_with_file_option(self)

    def assertGetHook(self, conf, name, value):
        calls = []

        def hook(*args):
            calls.append(args)
        config.OldConfigHooks.install_named_hook('get', hook, None)
        self.addCleanup(config.OldConfigHooks.uninstall_named_hook, 'get', None)
        self.assertLength(0, calls)
        actual_value = conf.get_user_option(name)
        self.assertEqual(value, actual_value)
        self.assertLength(1, calls)
        self.assertEqual((conf, name, value), calls[0])

    def test_get_hook_breezy(self):
        self.assertGetHook(self.breezy_config, 'file', 'breezy')

    def test_get_hook_locations(self):
        self.assertGetHook(self.locations_config, 'file', 'locations')

    def test_get_hook_branch(self):
        self.branch_config.set_user_option('file2', 'branch')
        self.assertGetHook(self.branch_config, 'file2', 'branch')

    def assertSetHook(self, conf, name, value):
        calls = []

        def hook(*args):
            calls.append(args)
        config.OldConfigHooks.install_named_hook('set', hook, None)
        self.addCleanup(config.OldConfigHooks.uninstall_named_hook, 'set', None)
        self.assertLength(0, calls)
        conf.set_user_option(name, value)
        self.assertLength(1, calls)
        self.assertEqual((name, value), calls[0][1:])

    def test_set_hook_breezy(self):
        self.assertSetHook(self.breezy_config, 'foo', 'breezy')

    def test_set_hook_locations(self):
        self.assertSetHook(self.locations_config, 'foo', 'locations')

    def test_set_hook_branch(self):
        self.assertSetHook(self.branch_config, 'foo', 'branch')

    def assertRemoveHook(self, conf, name, section_name=None):
        calls = []

        def hook(*args):
            calls.append(args)
        config.OldConfigHooks.install_named_hook('remove', hook, None)
        self.addCleanup(config.OldConfigHooks.uninstall_named_hook, 'remove', None)
        self.assertLength(0, calls)
        conf.remove_user_option(name, section_name)
        self.assertLength(1, calls)
        self.assertEqual((name,), calls[0][1:])

    def test_remove_hook_breezy(self):
        self.assertRemoveHook(self.breezy_config, 'file')

    def test_remove_hook_locations(self):
        self.assertRemoveHook(self.locations_config, 'file', self.locations_config.location)

    def test_remove_hook_branch(self):
        self.assertRemoveHook(self.branch_config, 'file')

    def assertLoadHook(self, name, conf_class, *conf_args):
        calls = []

        def hook(*args):
            calls.append(args)
        config.OldConfigHooks.install_named_hook('load', hook, None)
        self.addCleanup(config.OldConfigHooks.uninstall_named_hook, 'load', None)
        self.assertLength(0, calls)
        conf = conf_class(*conf_args)
        conf.get_user_option(name)
        self.assertLength(1, calls)

    def test_load_hook_breezy(self):
        self.assertLoadHook('file', config.GlobalConfig)

    def test_load_hook_locations(self):
        self.assertLoadHook('file', config.LocationConfig, self.tree.basedir)

    def test_load_hook_branch(self):
        self.assertLoadHook('file', config.BranchConfig, self.tree.branch)

    def assertSaveHook(self, conf):
        calls = []

        def hook(*args):
            calls.append(args)
        config.OldConfigHooks.install_named_hook('save', hook, None)
        self.addCleanup(config.OldConfigHooks.uninstall_named_hook, 'save', None)
        self.assertLength(0, calls)
        conf.set_user_option('foo', 'bar')
        self.assertLength(1, calls)

    def test_save_hook_breezy(self):
        self.assertSaveHook(self.breezy_config)

    def test_save_hook_locations(self):
        self.assertSaveHook(self.locations_config)

    def test_save_hook_branch(self):
        self.assertSaveHook(self.branch_config)