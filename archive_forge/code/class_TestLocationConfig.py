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
class TestLocationConfig(tests.TestCaseInTempDir, TestOptionsMixin):

    def test_constructs_valid(self):
        config.LocationConfig('http://example.com')

    def test_constructs_error(self):
        self.assertRaises(TypeError, config.LocationConfig)

    def test_branch_calls_read_filenames(self):
        oldparserclass = config.ConfigObj
        config.ConfigObj = InstrumentedConfigObj
        try:
            my_config = config.LocationConfig('http://www.example.com')
            parser = my_config._get_parser()
        finally:
            config.ConfigObj = oldparserclass
        self.assertIsInstance(parser, InstrumentedConfigObj)
        self.assertEqual(parser._calls, [('__init__', bedding.locations_config_path(), 'utf-8')])

    def test_get_global_config(self):
        my_config = config.BranchConfig(FakeBranch('http://example.com'))
        global_config = my_config._get_global_config()
        self.assertIsInstance(global_config, config.GlobalConfig)
        self.assertIs(global_config, my_config._get_global_config())

    def assertLocationMatching(self, expected):
        self.assertEqual(expected, list(self.my_location_config._get_matching_sections()))

    def test__get_matching_sections_no_match(self):
        self.get_branch_config('/')
        self.assertLocationMatching([])

    def test__get_matching_sections_exact(self):
        self.get_branch_config('http://www.example.com')
        self.assertLocationMatching([('http://www.example.com', '')])

    def test__get_matching_sections_suffix_does_not(self):
        self.get_branch_config('http://www.example.com-com')
        self.assertLocationMatching([])

    def test__get_matching_sections_subdir_recursive(self):
        self.get_branch_config('http://www.example.com/com')
        self.assertLocationMatching([('http://www.example.com', 'com')])

    def test__get_matching_sections_ignoreparent(self):
        self.get_branch_config('http://www.example.com/ignoreparent')
        self.assertLocationMatching([('http://www.example.com/ignoreparent', '')])

    def test__get_matching_sections_ignoreparent_subdir(self):
        self.get_branch_config('http://www.example.com/ignoreparent/childbranch')
        self.assertLocationMatching([('http://www.example.com/ignoreparent', 'childbranch')])

    def test__get_matching_sections_subdir_trailing_slash(self):
        self.get_branch_config('/b')
        self.assertLocationMatching([('/b/', '')])

    def test__get_matching_sections_subdir_child(self):
        self.get_branch_config('/a/foo')
        self.assertLocationMatching([('/a/*', ''), ('/a/', 'foo')])

    def test__get_matching_sections_subdir_child_child(self):
        self.get_branch_config('/a/foo/bar')
        self.assertLocationMatching([('/a/*', 'bar'), ('/a/', 'foo/bar')])

    def test__get_matching_sections_trailing_slash_with_children(self):
        self.get_branch_config('/a/')
        self.assertLocationMatching([('/a/', '')])

    def test__get_matching_sections_explicit_over_glob(self):
        self.get_branch_config('/a/c')
        self.assertLocationMatching([('/a/c', ''), ('/a/*', ''), ('/a/', 'c')])

    def test__get_option_policy_normal(self):
        self.get_branch_config('http://www.example.com')
        self.assertEqual(self.my_location_config._get_option_policy('http://www.example.com', 'normal_option'), config.POLICY_NONE)

    def test__get_option_policy_norecurse(self):
        self.get_branch_config('http://www.example.com')
        self.assertEqual(self.my_location_config._get_option_policy('http://www.example.com', 'norecurse_option'), config.POLICY_NORECURSE)
        self.assertEqual(self.my_location_config._get_option_policy('http://www.example.com/norecurse', 'normal_option'), config.POLICY_NORECURSE)

    def test__get_option_policy_normal_appendpath(self):
        self.get_branch_config('http://www.example.com')
        self.assertEqual(self.my_location_config._get_option_policy('http://www.example.com', 'appendpath_option'), config.POLICY_APPENDPATH)

    def test__get_options_with_policy(self):
        self.get_branch_config('/dir/subdir', location_config='[/dir]\nother_url = /other-dir\nother_url:policy = appendpath\n[/dir/subdir]\nother_url = /other-subdir\n')
        self.assertOptions([('other_url', '/other-subdir', '/dir/subdir', 'locations'), ('other_url', '/other-dir', '/dir', 'locations'), ('other_url:policy', 'appendpath', '/dir', 'locations')], self.my_location_config)

    def test_location_without_username(self):
        self.get_branch_config('http://www.example.com/ignoreparent')
        self.assertEqual('Erik Bågfors <erik@bagfors.nu>', self.my_config.username())

    def test_location_not_listed(self):
        """Test that the global username is used when no location matches"""
        self.get_branch_config('/home/robertc/sources')
        self.assertEqual('Erik Bågfors <erik@bagfors.nu>', self.my_config.username())

    def test_overriding_location(self):
        self.get_branch_config('http://www.example.com/foo')
        self.assertEqual('Robert Collins <robertc@example.org>', self.my_config.username())

    def test_get_user_option_global(self):
        self.get_branch_config('/a')
        self.assertEqual('something', self.my_config.get_user_option('user_global_option'))

    def test_get_user_option_local(self):
        self.get_branch_config('/a')
        self.assertEqual('local', self.my_config.get_user_option('user_local_option'))

    def test_get_user_option_appendpath(self):
        self.get_branch_config('http://www.example.com')
        self.assertEqual('append', self.my_config.get_user_option('appendpath_option'))
        self.get_branch_config('http://www.example.com/a/b/c')
        self.assertEqual('append/a/b/c', self.my_config.get_user_option('appendpath_option'))
        self.get_branch_config('http://www.example.com/dir/a/b/c')
        self.assertEqual('normal', self.my_config.get_user_option('appendpath_option'))

    def test_get_user_option_norecurse(self):
        self.get_branch_config('http://www.example.com')
        self.assertEqual('norecurse', self.my_config.get_user_option('norecurse_option'))
        self.get_branch_config('http://www.example.com/dir')
        self.assertEqual(None, self.my_config.get_user_option('norecurse_option'))
        self.get_branch_config('http://www.example.com/norecurse')
        self.assertEqual('norecurse', self.my_config.get_user_option('normal_option'))
        self.get_branch_config('http://www.example.com/norecurse/subdir')
        self.assertEqual('normal', self.my_config.get_user_option('normal_option'))

    def test_set_user_option_norecurse(self):
        self.get_branch_config('http://www.example.com')
        self.my_config.set_user_option('foo', 'bar', store=config.STORE_LOCATION_NORECURSE)
        self.assertEqual(self.my_location_config._get_option_policy('http://www.example.com', 'foo'), config.POLICY_NORECURSE)

    def test_set_user_option_appendpath(self):
        self.get_branch_config('http://www.example.com')
        self.my_config.set_user_option('foo', 'bar', store=config.STORE_LOCATION_APPENDPATH)
        self.assertEqual(self.my_location_config._get_option_policy('http://www.example.com', 'foo'), config.POLICY_APPENDPATH)

    def test_set_user_option_change_policy(self):
        self.get_branch_config('http://www.example.com')
        self.my_config.set_user_option('norecurse_option', 'normal', store=config.STORE_LOCATION)
        self.assertEqual(self.my_location_config._get_option_policy('http://www.example.com', 'norecurse_option'), config.POLICY_NONE)

    def get_branch_config(self, location, global_config=None, location_config=None):
        my_branch = FakeBranch(location)
        if global_config is None:
            global_config = sample_config_text
        if location_config is None:
            location_config = sample_branches_text
        config.GlobalConfig.from_string(global_config, save=True)
        config.LocationConfig.from_string(location_config, my_branch.base, save=True)
        my_config = config.BranchConfig(my_branch)
        self.my_config = my_config
        self.my_location_config = my_config._get_location_config()

    def test_set_user_setting_sets_and_saves2(self):
        self.get_branch_config('/a/c')
        self.assertIs(self.my_config.get_user_option('foo'), None)
        self.my_config.set_user_option('foo', 'bar')
        self.assertEqual(self.my_config.branch.control_files.files['branch.conf'].strip(), b'foo = bar')
        self.assertEqual(self.my_config.get_user_option('foo'), 'bar')
        self.my_config.set_user_option('foo', 'baz', store=config.STORE_LOCATION)
        self.assertEqual(self.my_config.get_user_option('foo'), 'baz')
        self.my_config.set_user_option('foo', 'qux')
        self.assertEqual(self.my_config.get_user_option('foo'), 'baz')

    def test_get_bzr_remote_path(self):
        my_config = config.LocationConfig('/a/c')
        self.assertEqual('bzr', my_config.get_bzr_remote_path())
        my_config.set_user_option('bzr_remote_path', '/path-bzr')
        self.assertEqual('/path-bzr', my_config.get_bzr_remote_path())
        self.overrideEnv('BZR_REMOTE_PATH', '/environ-bzr')
        self.assertEqual('/environ-bzr', my_config.get_bzr_remote_path())