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
class SadPathTestCase(BaseTestCase):

    def test_unknown_attr(self):
        self.conf([])
        self.assertFalse(hasattr(self.conf, 'foo'))
        self.assertRaises(AttributeError, getattr, self.conf, 'foo')
        self.assertRaises(cfg.NoSuchOptError, self.conf._get, 'foo')
        self.assertRaises(cfg.NoSuchOptError, self.conf.__getattr__, 'foo')

    def test_unknown_attr_is_attr_error(self):
        self.conf([])
        self.assertFalse(hasattr(self.conf, 'foo'))
        self.assertRaises(AttributeError, getattr, self.conf, 'foo')

    def test_unknown_group_attr(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertFalse(hasattr(self.conf.blaa, 'foo'))
        self.assertRaises(cfg.NoSuchOptError, getattr, self.conf.blaa, 'foo')

    def test_ok_duplicate(self):
        opt = cfg.StrOpt('foo')
        self.conf.register_cli_opt(opt)
        opt2 = cfg.StrOpt('foo')
        self.conf.register_cli_opt(opt2)
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertIsNone(self.conf.foo)

    def test_error_duplicate(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', help='bar'))
        self.assertRaises(cfg.DuplicateOptError, self.conf.register_cli_opt, cfg.StrOpt('foo'))

    def test_error_duplicate_with_different_dest(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', dest='f'))
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        self.assertRaises(cfg.DuplicateOptError, self.conf, [])

    def test_error_duplicate_short(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', short='f'))
        self.conf.register_cli_opt(cfg.StrOpt('bar', short='f'))
        self.assertRaises(cfg.DuplicateOptError, self.conf, [])

    def test_already_parsed(self):
        self.conf([])
        self.assertRaises(cfg.ArgsAlreadyParsedError, self.conf.register_cli_opt, cfg.StrOpt('foo'))

    def test_bad_cli_arg(self):
        self.conf.register_opt(cfg.BoolOpt('foo'))
        self.useFixture(fixtures.MonkeyPatch('sys.stderr', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--foo'])
        self.assertIn('error', sys.stderr.getvalue())
        self.assertIn('--foo', sys.stderr.getvalue())

    def _do_test_bad_cli_value(self, opt_class):
        self.conf.register_cli_opt(opt_class('foo'))
        self.useFixture(fixtures.MonkeyPatch('sys.stderr', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--foo', 'bar'])
        self.assertIn('foo', sys.stderr.getvalue())
        self.assertIn('bar', sys.stderr.getvalue())

    def test_bad_int_arg(self):
        self._do_test_bad_cli_value(cfg.IntOpt)

    def test_bad_float_arg(self):
        self._do_test_bad_cli_value(cfg.FloatOpt)

    def test_conf_file_not_found(self):
        fd, path = tempfile.mkstemp()
        os.remove(path)
        self.assertRaises(cfg.ConfigFilesNotFoundError, self.conf, ['--config-file', path])

    @unittest.skipIf(os.getuid() == 0, 'Not supported with the root privileges')
    def test_conf_file_permission_denied(self):
        fd, path = tempfile.mkstemp()
        os.chmod(path, 0)
        self.assertRaises(cfg.ConfigFilesPermissionDeniedError, self.conf, ['--config-file', path])
        os.remove(path)

    def test_conf_file_broken(self):
        paths = self.create_tempfiles([('test', 'foo')])
        self.assertRaises(cfg.ConfigFileParseError, self.conf, ['--config-file', paths[0]])

    def _do_test_conf_file_bad_value(self, opt_class):
        self.conf.register_opt(opt_class('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(ValueError, getattr, self.conf, 'foo')
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_bad_bool(self):
        self._do_test_conf_file_bad_value(cfg.BoolOpt)

    def test_conf_file_bad_int(self):
        self._do_test_conf_file_bad_value(cfg.IntOpt)

    def test_conf_file_bad_float(self):
        self._do_test_conf_file_bad_value(cfg.FloatOpt)

    def test_str_sub_none_value(self):
        self.conf.register_cli_opt(cfg.StrOpt('oo'))
        self.conf.register_cli_opt(cfg.StrOpt('bar', default='$oo'))
        self.conf.register_cli_opt(cfg.StrOpt('barbar', default='foo $oo foo'))
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'bar'))
        self.assertEqual('', self.conf.bar)
        self.assertEqual('foo  foo', self.conf.barbar)

    def test_str_sub_from_group(self):
        self.conf.register_group(cfg.OptGroup('f'))
        self.conf.register_cli_opt(cfg.StrOpt('oo', default='blaa'), group='f')
        self.conf.register_cli_opt(cfg.StrOpt('bar', default='$f.oo'))
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'bar'))
        self.assertEqual('blaa', self.conf.bar)

    def test_str_sub_from_group_with_brace(self):
        self.conf.register_group(cfg.OptGroup('f'))
        self.conf.register_cli_opt(cfg.StrOpt('oo', default='blaa'), group='f')
        self.conf.register_cli_opt(cfg.StrOpt('bar', default='${f.oo}'))
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'bar'))
        self.assertEqual('blaa', self.conf.bar)

    def test_set_default_unknown_attr(self):
        self.conf([])
        self.assertRaises(cfg.NoSuchOptError, self.conf.set_default, 'foo', 'bar')

    def test_set_default_unknown_group(self):
        self.conf([])
        self.assertRaises(cfg.NoSuchGroupError, self.conf.set_default, 'foo', 'bar', group='blaa')

    def test_set_override_unknown_attr(self):
        self.conf([])
        self.assertRaises(cfg.NoSuchOptError, self.conf.set_override, 'foo', 'bar')

    def test_set_override_unknown_group(self):
        self.conf([])
        self.assertRaises(cfg.NoSuchGroupError, self.conf.set_override, 'foo', 'bar', group='blaa')