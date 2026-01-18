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
class TestBranchConfig(tests.TestCaseWithTransport):

    def test_constructs_valid(self):
        branch = FakeBranch()
        my_config = config.BranchConfig(branch)
        self.assertIsNot(None, my_config)

    def test_constructs_error(self):
        self.assertRaises(TypeError, config.BranchConfig)

    def test_get_location_config(self):
        branch = FakeBranch()
        my_config = config.BranchConfig(branch)
        location_config = my_config._get_location_config()
        self.assertEqual(branch.base, location_config.location)
        self.assertIs(location_config, my_config._get_location_config())

    def test_get_config(self):
        """The Branch.get_config method works properly"""
        b = controldir.ControlDir.create_standalone_workingtree('.').branch
        my_config = b.get_config()
        self.assertIs(my_config.get_user_option('wacky'), None)
        my_config.set_user_option('wacky', 'unlikely')
        self.assertEqual(my_config.get_user_option('wacky'), 'unlikely')
        b2 = branch.Branch.open('.')
        my_config2 = b2.get_config()
        self.assertEqual(my_config2.get_user_option('wacky'), 'unlikely')

    def test_has_explicit_nickname(self):
        b = self.make_branch('.')
        self.assertFalse(b.get_config().has_explicit_nickname())
        b.nick = 'foo'
        self.assertTrue(b.get_config().has_explicit_nickname())

    def test_config_url(self):
        """The Branch.get_config will use section that uses a local url"""
        branch = self.make_branch('branch')
        self.assertEqual('branch', branch.nick)
        local_url = urlutils.local_path_to_url('branch')
        conf = config.LocationConfig.from_string('[{}]\nnickname = foobar'.format(local_url), local_url, save=True)
        self.assertIsNot(None, conf)
        self.assertEqual('foobar', branch.nick)

    def test_config_local_path(self):
        """The Branch.get_config will use a local system path"""
        branch = self.make_branch('branch')
        self.assertEqual('branch', branch.nick)
        local_path = osutils.getcwd().encode('utf8')
        config.LocationConfig.from_string(b'[%s/branch]\nnickname = barry' % (local_path,), 'branch', save=True)
        self.assertEqual('barry', branch.nick)

    def test_config_creates_local(self):
        """Creating a new entry in config uses a local path."""
        branch = self.make_branch('branch', format='knit')
        branch.set_push_location('http://foobar')
        local_path = osutils.getcwd().encode('utf8')
        self.check_file_contents(bedding.locations_config_path(), b'[%s/branch]\npush_location = http://foobar\npush_location:policy = norecurse\n' % (local_path,))

    def test_autonick_urlencoded(self):
        b = self.make_branch('!repo')
        self.assertEqual('!repo', b.get_config().get_nickname())

    def test_autonick_uses_branch_name(self):
        b = self.make_branch('foo', name='bar')
        self.assertEqual('bar', b.get_config().get_nickname())

    def test_warn_if_masked(self):
        warnings = []

        def warning(*args):
            warnings.append(args[0] % args[1:])
        self.overrideAttr(trace, 'warning', warning)

        def set_option(store, warn_masked=True):
            warnings[:] = []
            conf.set_user_option('example_option', repr(store), store=store, warn_masked=warn_masked)

        def assertWarning(warning):
            if warning is None:
                self.assertEqual(0, len(warnings))
            else:
                self.assertEqual(1, len(warnings))
                self.assertEqual(warning, warnings[0])
        branch = self.make_branch('.')
        conf = branch.get_config()
        set_option(config.STORE_GLOBAL)
        assertWarning(None)
        set_option(config.STORE_BRANCH)
        assertWarning(None)
        set_option(config.STORE_GLOBAL)
        assertWarning('Value "4" is masked by "3" from branch.conf')
        set_option(config.STORE_GLOBAL, warn_masked=False)
        assertWarning(None)
        set_option(config.STORE_LOCATION)
        assertWarning(None)
        set_option(config.STORE_BRANCH)
        assertWarning('Value "3" is masked by "0" from locations.conf')
        set_option(config.STORE_BRANCH, warn_masked=False)
        assertWarning(None)