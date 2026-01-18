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
class RequiredOptsTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.conf.register_opt(cfg.StrOpt('boo', required=False))

    def test_required_opt(self):
        self.conf.register_opt(cfg.StrOpt('foo', required=True))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)

    def test_required_cli_opt(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=True))
        self.conf(['--foo', 'bar'])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)

    def test_required_cli_opt_with_dash(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo-bar', required=True))
        self.conf(['--foo-bar', 'baz'])
        self.assertTrue(hasattr(self.conf, 'foo_bar'))
        self.assertEqual('baz', self.conf.foo_bar)

    def test_missing_required_opt(self):
        self.conf.register_opt(cfg.StrOpt('foo', required=True))
        self.assertRaises(cfg.RequiredOptError, self.conf, [])

    def test_missing_required_cli_opt(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=True))
        self.assertRaises(cfg.RequiredOptError, self.conf, [])

    def test_required_group_opt(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo', required=True), group='blaa')
        paths = self.create_tempfiles([('test', '[blaa]\nfoo = bar')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_required_cli_group_opt(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=True), group='blaa')
        self.conf(['--blaa-foo', 'bar'])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_missing_required_group_opt(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo', required=True), group='blaa')
        self.assertRaises(cfg.RequiredOptError, self.conf, [])

    def test_missing_required_cli_group_opt(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=True), group='blaa')
        self.assertRaises(cfg.RequiredOptError, self.conf, [])

    def test_required_opt_with_default(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=True))
        self.conf.set_default('foo', 'bar')
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)

    def test_required_opt_with_override(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=True))
        self.conf.set_override('foo', 'bar')
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)