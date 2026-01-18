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
class DefaultConfigDirsTestCase(BaseTestCase):

    def test_use_default(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('foo.conf.d/foo', '[DEFAULT]\nfoo = bar\n')])
        p = os.path.dirname(paths[0])
        self.conf.register_cli_opt(cfg.StrOpt('config-dir-foo'))
        self.conf(args=['--config-dir-foo', 'foo.conf.d'], default_config_dirs=[p])
        self.assertEqual([p], self.conf.config_dir)
        self.assertEqual('bar', self.conf.foo)

    def test_do_not_use_default_multi_arg(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('foo.conf.d/foo', '[DEFAULT]\nfoo = bar\n')])
        p = os.path.dirname(paths[0])
        self.conf(args=['--config-dir', p], default_config_dirs=['bar.conf.d'])
        self.assertEqual([p], self.conf.config_dirs)
        self.assertEqual('bar', self.conf.foo)

    def test_do_not_use_default_single_arg(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('foo.conf.d/foo', '[DEFAULT]\nfoo = bar\n')])
        p = os.path.dirname(paths[0])
        self.conf(args=['--config-dir=' + p], default_config_dirs=['bar.conf.d'])
        self.assertEqual([p], self.conf.config_dir)
        self.assertEqual('bar', self.conf.foo)

    def test_no_default_config_dir(self):
        self.conf(args=[])
        self.assertEqual([], self.conf.config_dir)

    def test_find_default_config_dir(self):
        paths = self.create_tempfiles([('def.conf.d/def', '[DEFAULT]')])
        p = os.path.dirname(paths[0])
        self.useFixture(fixtures.MonkeyPatch('oslo_config.cfg.find_config_dirs', lambda project, prog: p))
        self.conf(args=[], default_config_dirs=None)
        self.assertEqual([p], self.conf.config_dir)

    def test_default_config_dir(self):
        paths = self.create_tempfiles([('def.conf.d/def', '[DEFAULT]')])
        p = os.path.dirname(paths[0])
        self.conf(args=[], default_config_dirs=[p])
        self.assertEqual([p], self.conf.config_dir)

    def test_default_config_dir_with_value(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('def.conf.d/def', '[DEFAULT]\nfoo = bar\n')])
        p = os.path.dirname(paths[0])
        self.conf(args=[], default_config_dirs=[p])
        self.assertEqual([p], self.conf.config_dir)
        self.assertEqual('bar', self.conf.foo)

    def test_default_config_dir_priority(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('def.conf.d/def', '[DEFAULT]\nfoo = bar\n')])
        p = os.path.dirname(paths[0])
        self.conf(args=['--foo=blaa'], default_config_dirs=[p])
        self.assertEqual([p], self.conf.config_dir)
        self.assertEqual('blaa', self.conf.foo)