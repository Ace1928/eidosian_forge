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
class LogLevelTestCase(BaseTestCase):

    def setUp(self):
        super(LogLevelTestCase, self).setUp()
        levels = self.CONF.default_log_levels
        info_level = 'nova-test'
        warn_level = 'nova-not-debug'
        other_level = 'nova-below-debug'
        trace_level = 'nova-trace'
        levels.append(info_level + '=INFO')
        levels.append(warn_level + '=WARN')
        levels.append(other_level + '=7')
        levels.append(trace_level + '=TRACE')
        self.config(default_log_levels=levels)
        log.setup(self.CONF, 'testing')
        self.log = log.getLogger(info_level)
        self.log_no_debug = log.getLogger(warn_level)
        self.log_below_debug = log.getLogger(other_level)
        self.log_trace = log.getLogger(trace_level)

    def test_is_enabled_for(self):
        self.assertTrue(self.log.isEnabledFor(logging.INFO))
        self.assertFalse(self.log_no_debug.isEnabledFor(logging.DEBUG))
        self.assertTrue(self.log_below_debug.isEnabledFor(logging.DEBUG))
        self.assertTrue(self.log_below_debug.isEnabledFor(7))
        self.assertTrue(self.log_trace.isEnabledFor(log.TRACE))

    def test_has_level_from_flags(self):
        self.assertEqual(logging.INFO, self.log.logger.getEffectiveLevel())

    def test_has_level_from_flags_for_trace(self):
        self.assertEqual(log.TRACE, self.log_trace.logger.getEffectiveLevel())

    def test_child_log_has_level_of_parent_flag(self):
        logger = log.getLogger('nova-test.foo')
        self.assertEqual(logging.INFO, logger.logger.getEffectiveLevel())

    def test_child_log_has_level_of_parent_flag_for_trace(self):
        logger = log.getLogger('nova-trace.foo')
        self.assertEqual(log.TRACE, logger.logger.getEffectiveLevel())

    def test_get_loggers(self):
        log._loggers['sentinel_log'] = mock.sentinel.sentinel_log
        res = log.get_loggers()
        self.assertDictEqual(log._loggers, res)