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
class ChoicesTestCase(BaseTestCase):

    def test_choice_default(self):
        self.conf.register_cli_opt(cfg.StrOpt('protocol', default='http', choices=['http', 'https', 'ftp']))
        self.conf([])
        self.assertEqual('http', self.conf.protocol)

    def test_choice_good(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', choices=['bar1', 'bar2']))
        self.conf(['--foo', 'bar1'])
        self.assertEqual('bar1', self.conf.foo)

    def test_choice_bad(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', choices=['bar1', 'bar2']))
        self.assertRaises(SystemExit, self.conf, ['--foo', 'bar3'])

    def test_conf_file_choice_value(self):
        self.conf.register_opt(cfg.StrOpt('foo', choices=['bar1', 'bar2']))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar1\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar1', self.conf.foo)

    def test_conf_file_choice_empty_value(self):
        self.conf.register_opt(cfg.StrOpt('foo', choices=['', 'bar1', 'bar2']))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = \n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('', self.conf.foo)

    def test_conf_file_choice_none_value(self):
        self.conf.register_opt(cfg.StrOpt('foo', default=None, choices=[None, 'bar1', 'bar2']))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertIsNone(self.conf.foo)

    def test_conf_file_bad_choice_value(self):
        self.conf.register_opt(cfg.StrOpt('foo', choices=['bar1', 'bar2']))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar3\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')
        self.assertRaises(ValueError, getattr, self.conf, 'foo')

    def test_conf_file_choice_value_override(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', choices=['baar', 'baaar']))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = baar\n'), ('2', '[DEFAULT]\nfoo = baaar\n')])
        self.conf(['--foo', 'baar', '--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('baaar', self.conf.foo)

    def test_conf_file_choice_bad_default(self):
        self.assertRaises(cfg.DefaultValueError, cfg.StrOpt, 'foo', choices=['baar', 'baaar'], default='foobaz')