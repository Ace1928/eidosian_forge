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
class TestOldConfigHooksForRemote(tests.TestCaseWithTransport):
    """Tests config hooks for remote configs.

    No tests for the remove hook as this is not implemented there.
    """

    def setUp(self):
        super().setUp()
        self.transport_server = test_server.SmartTCPServer_for_testing
        create_configs_with_file_option(self)

    def assertGetHook(self, conf, name, value):
        calls = []

        def hook(*args):
            calls.append(args)
        config.OldConfigHooks.install_named_hook('get', hook, None)
        self.addCleanup(config.OldConfigHooks.uninstall_named_hook, 'get', None)
        self.assertLength(0, calls)
        actual_value = conf.get_option(name)
        self.assertEqual(value, actual_value)
        self.assertLength(1, calls)
        self.assertEqual((conf, name, value), calls[0])

    def test_get_hook_remote_branch(self):
        remote_branch = branch.Branch.open(self.get_url('tree'))
        self.assertGetHook(remote_branch._get_config(), 'file', 'branch')

    def test_get_hook_remote_bzrdir(self):
        remote_bzrdir = controldir.ControlDir.open(self.get_url('tree'))
        conf = remote_bzrdir._get_config()
        conf.set_option('remotedir', 'file')
        self.assertGetHook(conf, 'file', 'remotedir')

    def assertSetHook(self, conf, name, value):
        calls = []

        def hook(*args):
            calls.append(args)
        config.OldConfigHooks.install_named_hook('set', hook, None)
        self.addCleanup(config.OldConfigHooks.uninstall_named_hook, 'set', None)
        self.assertLength(0, calls)
        conf.set_option(value, name)
        self.assertLength(1, calls)
        self.assertEqual((name, value), calls[0][1:])

    def test_set_hook_remote_branch(self):
        remote_branch = branch.Branch.open(self.get_url('tree'))
        self.addCleanup(remote_branch.lock_write().unlock)
        self.assertSetHook(remote_branch._get_config(), 'file', 'remote')

    def test_set_hook_remote_bzrdir(self):
        remote_branch = branch.Branch.open(self.get_url('tree'))
        self.addCleanup(remote_branch.lock_write().unlock)
        remote_bzrdir = controldir.ControlDir.open(self.get_url('tree'))
        self.assertSetHook(remote_bzrdir._get_config(), 'file', 'remotedir')

    def assertLoadHook(self, expected_nb_calls, name, conf_class, *conf_args):
        calls = []

        def hook(*args):
            calls.append(args)
        config.OldConfigHooks.install_named_hook('load', hook, None)
        self.addCleanup(config.OldConfigHooks.uninstall_named_hook, 'load', None)
        self.assertLength(0, calls)
        conf = conf_class(*conf_args)
        conf.get_option(name)
        self.assertLength(expected_nb_calls, calls)

    def test_load_hook_remote_branch(self):
        remote_branch = branch.Branch.open(self.get_url('tree'))
        self.assertLoadHook(1, 'file', remote.RemoteBranchConfig, remote_branch)

    def test_load_hook_remote_bzrdir(self):
        remote_bzrdir = controldir.ControlDir.open(self.get_url('tree'))
        conf = remote_bzrdir._get_config()
        conf.set_option('remotedir', 'file')
        self.assertLoadHook(2, 'file', remote.RemoteBzrDirConfig, remote_bzrdir)

    def assertSaveHook(self, conf):
        calls = []

        def hook(*args):
            calls.append(args)
        config.OldConfigHooks.install_named_hook('save', hook, None)
        self.addCleanup(config.OldConfigHooks.uninstall_named_hook, 'save', None)
        self.assertLength(0, calls)
        conf.set_option('foo', 'bar')
        self.assertLength(1, calls)

    def test_save_hook_remote_branch(self):
        remote_branch = branch.Branch.open(self.get_url('tree'))
        self.addCleanup(remote_branch.lock_write().unlock)
        self.assertSaveHook(remote_branch._get_config())

    def test_save_hook_remote_bzrdir(self):
        remote_branch = branch.Branch.open(self.get_url('tree'))
        self.addCleanup(remote_branch.lock_write().unlock)
        remote_bzrdir = controldir.ControlDir.open(self.get_url('tree'))
        self.assertSaveHook(remote_bzrdir._get_config())