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
class ImportGroupTestCase(BaseTestCase):

    def test_import_group(self):
        self.assertFalse(hasattr(cfg.CONF, 'qux'))
        cfg.CONF.import_group('qux', 'oslo_config.tests.testmods.baz_qux_opt')
        self.assertTrue(hasattr(cfg.CONF, 'qux'))
        self.assertTrue(hasattr(cfg.CONF.qux, 'baz'))

    def test_import_group_import_error(self):
        self.assertRaises(ImportError, cfg.CONF.import_group, 'qux', 'oslo_config.tests.testmods.bazzz_quxxx_opt')

    def test_import_group_no_such_group(self):
        self.assertRaises(cfg.NoSuchGroupError, cfg.CONF.import_group, 'quxxx', 'oslo_config.tests.testmods.baz_qux_opt')