from contextlib import contextmanager
import copy
import datetime
import io
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from unittest import mock
from dateutil import tz
from oslo_config import cfg
from oslo_config import fixture as fixture_config  # noqa
from oslo_context import context
from oslo_context import fixture as fixture_context
from oslo_i18n import fixture as fixture_trans
from oslo_serialization import jsonutils
from oslotest import base as test_base
import testtools
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
from oslo_log import log
from oslo_utils import units
class SetDefaultsTestCase(BaseTestCase):

    class TestConfigOpts(cfg.ConfigOpts):

        def __call__(self, args=None):
            return cfg.ConfigOpts.__call__(self, args=args, prog='test', version='1.0', usage='%(prog)s FOO BAR', default_config_files=[])

    def setUp(self):
        super(SetDefaultsTestCase, self).setUp()
        self.conf = self.TestConfigOpts()
        self.conf.register_opts(_options.log_opts)
        self.conf.register_cli_opts(_options.logging_cli_opts)
        self._orig_defaults = dict([(o.dest, o.default) for o in _options.log_opts])
        self.addCleanup(self._restore_log_defaults)

    def _restore_log_defaults(self):
        for opt in _options.log_opts:
            opt.default = self._orig_defaults[opt.dest]

    def test_default_log_level_to_none(self):
        log.set_defaults(logging_context_format_string=None, default_log_levels=None)
        self.conf([])
        self.assertEqual(_options.DEFAULT_LOG_LEVELS, self.conf.default_log_levels)

    def test_default_log_level_method(self):
        self.assertEqual(_options.DEFAULT_LOG_LEVELS, log.get_default_log_levels())

    def test_change_default(self):
        my_default = '%(asctime)s %(levelname)s %(name)s [%(request_id)s %(user_id)s %(project)s] %(instance)s%(message)s'
        log.set_defaults(logging_context_format_string=my_default)
        self.conf([])
        self.assertEqual(self.conf.logging_context_format_string, my_default)

    def test_change_default_log_level(self):
        package_log_level = 'foo=bar'
        log.set_defaults(default_log_levels=[package_log_level])
        self.conf([])
        self.assertEqual([package_log_level], self.conf.default_log_levels)
        self.assertIsNotNone(self.conf.logging_context_format_string)

    def test_tempest_set_log_file(self):
        log_file = 'foo.log'
        log.tempest_set_log_file(log_file)
        self.addCleanup(log.tempest_set_log_file, None)
        log.set_defaults()
        self.conf([])
        self.assertEqual(log_file, self.conf.log_file)

    def test_log_file_defaults_to_none(self):
        log.set_defaults()
        self.conf([])
        self.assertIsNone(self.conf.log_file)