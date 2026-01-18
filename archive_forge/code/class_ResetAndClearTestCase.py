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
class ResetAndClearTestCase(BaseTestCase):

    def test_clear(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        self.conf.register_cli_opt(cfg.StrOpt('bar'), group='blaa')
        self.assertIsNone(self.conf.foo)
        self.assertIsNone(self.conf.blaa.bar)
        self.conf(['--foo', 'foo', '--blaa-bar', 'bar'])
        self.assertEqual('foo', self.conf.foo)
        self.assertEqual('bar', self.conf.blaa.bar)
        self.conf.clear()
        self.assertIsNone(self.conf.foo)
        self.assertIsNone(self.conf.blaa.bar)

    def test_reset_and_clear_with_defaults_and_overrides(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        self.conf.register_cli_opt(cfg.StrOpt('bar'), group='blaa')
        self.conf.set_default('foo', 'foo')
        self.conf.set_override('bar', 'bar', group='blaa')
        self.conf(['--foo', 'foofoo'])
        self.assertEqual('foofoo', self.conf.foo)
        self.assertEqual('bar', self.conf.blaa.bar)
        self.conf.clear()
        self.assertEqual('foo', self.conf.foo)
        self.assertEqual('bar', self.conf.blaa.bar)
        self.conf.reset()
        self.assertIsNone(self.conf.foo)
        self.assertIsNone(self.conf.blaa.bar)