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
class ImportOptTestCase(BaseTestCase):

    def test_import_opt(self):
        self.assertFalse(hasattr(cfg.CONF, 'blaa'))
        cfg.CONF.import_opt('blaa', 'oslo_config.tests.testmods.blaa_opt')
        self.assertTrue(hasattr(cfg.CONF, 'blaa'))

    def test_import_opt_in_group(self):
        self.assertFalse(hasattr(cfg.CONF, 'bar'))
        cfg.CONF.import_opt('foo', 'oslo_config.tests.testmods.bar_foo_opt', group='bar')
        self.assertTrue(hasattr(cfg.CONF, 'bar'))
        self.assertTrue(hasattr(cfg.CONF.bar, 'foo'))

    def test_import_opt_import_errror(self):
        self.assertRaises(ImportError, cfg.CONF.import_opt, 'blaa', 'oslo_config.tests.testmods.blaablaa_opt')

    def test_import_opt_no_such_opt(self):
        self.assertRaises(cfg.NoSuchOptError, cfg.CONF.import_opt, 'blaablaa', 'oslo_config.tests.testmods.blaa_opt')

    def test_import_opt_no_such_group(self):
        self.assertRaises(cfg.NoSuchGroupError, cfg.CONF.import_opt, 'blaa', 'oslo_config.tests.testmods.blaa_opt', group='blaa')