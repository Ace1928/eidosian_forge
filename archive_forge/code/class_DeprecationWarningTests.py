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
@mock.patch('oslo_log.versionutils.report_deprecated_feature', _fake_deprecated_feature)
class DeprecationWarningTests(DeprecationWarningTestBase):
    log_prefix = 'Deprecated: '

    def test_DeprecatedOpt(self):
        default_deprecated = [cfg.DeprecatedOpt('bar')]
        other_deprecated = [cfg.DeprecatedOpt('baz', group='other')]
        self.conf.register_group(cfg.OptGroup('other'))
        self.conf.register_opt(cfg.StrOpt('foo', deprecated_opts=default_deprecated))
        self.conf.register_opt(cfg.StrOpt('foo', deprecated_opts=other_deprecated), group='other')
        paths = self.create_tempfiles([('test', '[DEFAULT]\n' + 'bar=baz\n' + '[other]\n' + 'baz=baz\n')])
        self.conf(['--config-file', paths[0]])
        self.assertEqual('baz', self.conf.foo)
        self.assertEqual('baz', self.conf.other.foo)
        self.assertIn('Option "bar" from group "DEFAULT"', self.log_fixture.output)
        self.assertIn('Option "baz" from group "other"', self.log_fixture.output)

    def test_check_deprecated(self):
        namespace = cfg._Namespace(None)
        deprecated_list = [('DEFAULT', 'bar')]
        namespace._check_deprecated(('DEFAULT', 'bar'), (None, 'foo'), deprecated_list)
        self.assert_message_logged('bar', 'DEFAULT', 'foo', 'DEFAULT')

    def assert_message_logged(self, deprecated_name, deprecated_group, current_name, current_group):
        expected = cfg._Namespace._deprecated_opt_message % {'dep_option': deprecated_name, 'dep_group': deprecated_group, 'option': current_name, 'group': current_group}
        self.assertIn(expected + '\n', self.log_fixture.output)

    def test_deprecated_for_removal(self):
        self.conf.register_opt(cfg.StrOpt('foo', deprecated_for_removal=True))
        self.conf.register_opt(cfg.StrOpt('bar', deprecated_for_removal=True))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n' + 'foo=bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertEqual('bar', self.conf.foo)
        self.assertEqual('bar', self.conf.foo)
        self.assertIsNone(self.conf.bar)
        expected = 'Option "foo" from group "DEFAULT" is deprecated for removal.  Its value may be silently ignored in the future.\n'
        self.assertIn(expected, self.log_fixture.output)

    def test_deprecated_for_removal_with_group(self):
        self.conf.register_group(cfg.OptGroup('other'))
        self.conf.register_opt(cfg.StrOpt('foo', deprecated_for_removal=True), group='other')
        self.conf.register_opt(cfg.StrOpt('bar', deprecated_for_removal=True), group='other')
        paths = self.create_tempfiles([('test', '[other]\n' + 'foo=bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertEqual('bar', self.conf.other.foo)
        self.assertEqual('bar', self.conf.other.foo)
        self.assertIsNone(self.conf.other.bar)
        expected = 'Option "foo" from group "other" is deprecated for removal.  Its value may be silently ignored in the future.\n'
        self.assertIn(expected, self.log_fixture.output)

    def test_deprecated_with_dest(self):
        self.conf.register_group(cfg.OptGroup('other'))
        self.conf.register_opt(cfg.StrOpt('foo-bar', deprecated_name='bar', dest='foo'), group='other')
        content = 'bar=baz'
        paths = self.create_tempfiles([('test', '[other]\n' + content + '\n')])
        self.conf(['--config-file', paths[0]])
        self.assertEqual('baz', self.conf.other.foo)
        expected = cfg._Namespace._deprecated_opt_message % {'dep_option': 'bar', 'dep_group': 'other', 'option': 'foo-bar', 'group': 'other'} + '\n'
        self.assertIn(expected, self.log_fixture.output)