import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
class ConfigFileMutateTestCase(BaseTestCase):

    def setUp(self):
        super(ConfigFileMutateTestCase, self).setUp()
        self.my_group = cfg.OptGroup('group', 'group options')
        self.conf.register_group(self.my_group)

    def _test_conf_files_mutate(self):
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = old_foo\n[group]\nboo = old_boo\n'), ('2', '[DEFAULT]\nfoo = new_foo\n[group]\nboo = new_boo\n')])
        self.conf(['--config-file', paths[0]])
        shutil.copy(paths[1], paths[0])
        return self.conf.mutate_config_files()

    def test_conf_files_mutate_none(self):
        """Test that immutable opts are not reloaded"""
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        self._test_conf_files_mutate()
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('old_foo', self.conf.foo)

    def test_conf_files_mutate_foo(self):
        """Test that a mutable opt can be reloaded."""
        self.conf.register_cli_opt(cfg.StrOpt('foo', mutable=True))
        self._test_conf_files_mutate()
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('new_foo', self.conf.foo)

    def test_conf_files_mutate_group(self):
        """Test that mutable opts in groups can be reloaded."""
        self.conf.register_cli_opt(cfg.StrOpt('boo', mutable=True), group=self.my_group)
        self._test_conf_files_mutate()
        self.assertTrue(hasattr(self.conf, 'group'))
        self.assertTrue(hasattr(self.conf.group, 'boo'))
        self.assertEqual('new_boo', self.conf.group.boo)

    def test_warn_immutability(self):
        self.log_fixture = self.useFixture(fixtures.FakeLogger())
        self.conf.register_cli_opt(cfg.StrOpt('foo', mutable=True))
        self.conf.register_cli_opt(cfg.StrOpt('boo'), group=self.my_group)
        self._test_conf_files_mutate()
        self.assertEqual('Ignoring change to immutable option group.boo\nOption DEFAULT.foo changed from [old_foo] to [new_foo]\n', self.log_fixture.output)

    def test_diff(self):
        self.log_fixture = self.useFixture(fixtures.FakeLogger())
        self.conf.register_cli_opt(cfg.StrOpt('imm'))
        self.conf.register_cli_opt(cfg.StrOpt('blank', mutable=True))
        self.conf.register_cli_opt(cfg.StrOpt('foo', mutable=True))
        self.conf.register_cli_opt(cfg.StrOpt('boo', mutable=True), group=self.my_group)
        diff = self._test_conf_files_mutate()
        self.assertEqual({(None, 'foo'): ('old_foo', 'new_foo'), ('group', 'boo'): ('old_boo', 'new_boo')}, diff)
        expected = 'Option DEFAULT.foo changed from [old_foo] to [new_foo]\nOption group.boo changed from [old_boo] to [new_boo]\n'
        self.assertEqual(expected, self.log_fixture.output)

    def test_hooks_invoked_once(self):
        fresh = {}
        result = [0]

        def foo(conf, foo_fresh):
            self.assertEqual(conf, self.conf)
            self.assertEqual(fresh, foo_fresh)
            result[0] += 1
        self.conf.register_mutate_hook(foo)
        self.conf.register_mutate_hook(foo)
        self._test_conf_files_mutate()
        self.assertEqual(1, result[0])

    def test_hooks_see_new_values(self):

        def foo(conf, fresh):
            self.assertEqual('new_foo', conf.foo)
        self.conf.register_cli_opt(cfg.StrOpt('foo', mutable=True))
        self.conf.register_mutate_hook(foo)
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = old_foo\n[group]\nboo = old_boo\n'), ('2', '[DEFAULT]\nfoo = new_foo\n[group]\nboo = new_boo\n')])
        self.conf(['--config-file', paths[0]])
        self.assertEqual('old_foo', self.conf.foo)
        shutil.copy(paths[1], paths[0])
        self.conf.mutate_config_files()
        self.assertEqual('new_foo', self.conf.foo)

    def test_clear(self):
        """Show that #clear doesn't undeclare opts.

        This justifies not clearing mutate_hooks either. ResetAndClearTestCase
        shows that values are cleared.
        """
        self.conf.register_cli_opt(cfg.StrOpt('cli'))
        self.conf.register_opt(cfg.StrOpt('foo'))
        dests = [info['opt'].dest for info, _ in self.conf._all_opt_infos()]
        self.assertIn('cli', dests)
        self.assertIn('foo', dests)
        self.conf.clear()
        dests = [info['opt'].dest for info, _ in self.conf._all_opt_infos()]
        self.assertIn('cli', dests)
        self.assertIn('foo', dests)