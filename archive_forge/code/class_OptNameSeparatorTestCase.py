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
class OptNameSeparatorTestCase(BaseTestCase):
    scenarios = [('hyphen', dict(opt_name='foo-bar', opt_dest='foo_bar', broken_opt_dest='foo-bar', cf_name='foo_bar', broken_cf_name='foo-bar', cli_name='foo-bar', hyphen=True)), ('underscore', dict(opt_name='foo_bar', opt_dest='foo_bar', broken_opt_dest='foo-bar', cf_name='foo_bar', broken_cf_name='foo-bar', cli_name='foo_bar', hyphen=False))]

    def test_attribute_and_key_name(self):
        self.conf.register_opt(cfg.StrOpt(self.opt_name))
        self.assertTrue(hasattr(self.conf, self.opt_dest))
        self.assertFalse(hasattr(self.conf, self.broken_opt_dest))
        self.assertIn(self.opt_dest, self.conf)
        self.assertNotIn(self.broken_opt_dest, self.conf)

    def test_cli_opt_name(self):
        self.conf.register_cli_opt(cfg.BoolOpt(self.opt_name))
        self.conf(['--' + self.cli_name])
        self.assertTrue(getattr(self.conf, self.opt_dest))

    def test_config_file_opt_name(self):
        self.conf.register_opt(cfg.BoolOpt(self.opt_name))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n' + self.cf_name + ' = True\n' + self.broken_cf_name + ' = False\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(getattr(self.conf, self.opt_dest))

    def test_deprecated_name(self):
        self.conf.register_opt(cfg.StrOpt('foobar', deprecated_name=self.opt_name))
        self.assertTrue(hasattr(self.conf, 'foobar'))
        self.assertTrue(hasattr(self.conf, self.opt_dest))
        self.assertFalse(hasattr(self.conf, self.broken_opt_dest))
        self.assertIn('foobar', self.conf)
        self.assertNotIn(self.opt_dest, self.conf)
        self.assertNotIn(self.broken_opt_dest, self.conf)

    def test_deprecated_name_alternate_group(self):
        self.conf.register_opt(cfg.StrOpt('foobar', deprecated_name=self.opt_name, deprecated_group='testing'), group=cfg.OptGroup('testing'))
        self.assertTrue(hasattr(self.conf.testing, 'foobar'))
        self.assertTrue(hasattr(self.conf.testing, self.opt_dest))
        self.assertFalse(hasattr(self.conf.testing, self.broken_opt_dest))
        self.assertIn('foobar', self.conf.testing)
        self.assertNotIn(self.opt_dest, self.conf.testing)
        self.assertNotIn(self.broken_opt_dest, self.conf.testing)

    def test_deprecated_name_cli(self):
        self.conf.register_cli_opt(cfg.BoolOpt('foobar', deprecated_name=self.opt_name))
        self.conf(['--' + self.cli_name])
        self.assertTrue(self.conf.foobar)

    def test_deprecated_name_config_file(self):
        self.conf.register_opt(cfg.BoolOpt('foobar', deprecated_name=self.opt_name))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n' + self.cf_name + ' = True\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(self.conf.foobar)

    def test_deprecated_opts(self):
        oldopts = [cfg.DeprecatedOpt(self.opt_name)]
        self.conf.register_opt(cfg.StrOpt('foobar', deprecated_opts=oldopts))
        self.assertTrue(hasattr(self.conf, 'foobar'))
        self.assertTrue(hasattr(self.conf, self.opt_dest))
        self.assertFalse(hasattr(self.conf, self.broken_opt_dest))
        self.assertIn('foobar', self.conf)
        self.assertNotIn(self.opt_dest, self.conf)
        self.assertNotIn(self.broken_opt_dest, self.conf)

    def test_deprecated_opts_cli(self):
        oldopts = [cfg.DeprecatedOpt(self.opt_name)]
        self.conf.register_cli_opt(cfg.BoolOpt('foobar', deprecated_opts=oldopts))
        self.conf(['--' + self.cli_name])
        self.assertTrue(self.conf.foobar)

    def test_deprecated_opts_config_file(self):
        oldopts = [cfg.DeprecatedOpt(self.opt_name)]
        self.conf.register_opt(cfg.BoolOpt('foobar', deprecated_opts=oldopts))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n' + self.cf_name + ' = True\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(self.conf.foobar)