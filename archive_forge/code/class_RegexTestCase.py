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
class RegexTestCase(BaseTestCase):

    def test_regex_good(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', regex='foo|bar'))
        self.conf(['--foo', 'bar'])
        self.assertEqual('bar', self.conf.foo)
        self.conf(['--foo', 'foo'])
        self.assertEqual('foo', self.conf.foo)
        self.conf(['--foo', 'foobar'])
        self.assertEqual('foobar', self.conf.foo)

    def test_regex_bad(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', regex='bar'))
        self.assertRaises(SystemExit, self.conf, ['--foo', 'foo'])

    def test_conf_file_regex_value(self):
        self.conf.register_opt(cfg.StrOpt('foo', regex='bar'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)

    def test_conf_file_regex_bad_value(self):
        self.conf.register_opt(cfg.StrOpt('foo', regex='bar'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = other\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaisesRegex(cfg.ConfigFileValueError, "doesn't match regex", self.conf._get, 'foo')
        self.assertRaisesRegex(ValueError, "doesn't match regex", getattr, self.conf, 'foo')

    def test_regex_with_choice(self):
        self.assertRaises(ValueError, cfg.StrOpt, 'foo', choices=['bar1'], regex='bar2')