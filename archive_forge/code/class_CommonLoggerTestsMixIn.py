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
class CommonLoggerTestsMixIn(object):
    """These tests are shared between LoggerTestCase and
    LazyLoggerTestCase.
    """

    def setUp(self):
        super(CommonLoggerTestsMixIn, self).setUp()
        self.config_fixture = self.useFixture(fixture_config.Config(cfg.ConfigOpts()))
        self.config = self.config_fixture.config
        self.CONF = self.config_fixture.conf
        log.register_options(self.config_fixture.conf)
        self.config(logging_context_format_string='%(asctime)s %(levelname)s %(name)s [%(request_id)s %(user)s %(project)s] %(message)s')
        self.log = None
        log._setup_logging_from_conf(self.config_fixture.conf, 'test', 'test')
        self.log_handlers = log.getLogger(None).logger.handlers

    def test_handlers_have_context_formatter(self):
        formatters_list = []
        for h in self.log.logger.handlers:
            f = h.formatter
            if isinstance(f, formatters.ContextFormatter):
                formatters_list.append(f)
        self.assertTrue(formatters_list)
        self.assertEqual(len(formatters_list), len(self.log.logger.handlers))

    def test_handles_context_kwarg(self):
        self.log.info('foo', context=_fake_context())
        self.assertTrue(True)

    def test_will_be_debug_if_debug_flag_set(self):
        self.config(debug=True)
        logger_name = 'test_is_debug'
        log.setup(self.CONF, logger_name)
        logger = logging.getLogger(logger_name)
        self.assertEqual(logging.DEBUG, logger.getEffectiveLevel())

    def test_will_be_info_if_debug_flag_not_set(self):
        self.config(debug=False)
        logger_name = 'test_is_not_debug'
        log.setup(self.CONF, logger_name)
        logger = logging.getLogger(logger_name)
        self.assertEqual(logging.INFO, logger.getEffectiveLevel())

    def test_no_logging_via_module(self):
        for func in ('critical', 'error', 'exception', 'warning', 'warn', 'info', 'debug', 'log'):
            self.assertRaises(AttributeError, getattr, log, func)

    @mock.patch('platform.system', return_value='Linux')
    def test_eventlog_missing(self, platform_mock):
        self.config(use_eventlog=True)
        self.assertRaises(RuntimeError, log._setup_logging_from_conf, self.CONF, 'test', 'test')

    @mock.patch('platform.system', return_value='Windows')
    @mock.patch('logging.handlers.NTEventLogHandler')
    @mock.patch('oslo_log.log.getLogger')
    def test_eventlog(self, loggers_mock, handler_mock, platform_mock):
        self.config(use_eventlog=True)
        log._setup_logging_from_conf(self.CONF, 'test', 'test')
        handler_mock.assert_called_once_with('test')
        mock_logger = loggers_mock.return_value.logger
        mock_logger.addHandler.assert_any_call(handler_mock.return_value)

    @mock.patch('oslo_log.watchers.FastWatchedFileHandler')
    @mock.patch('oslo_log.log._get_log_file_path', return_value='test.conf')
    @mock.patch('platform.system', return_value='Linux')
    def test_watchlog_on_linux(self, platfotm_mock, path_mock, handler_mock):
        self.config(watch_log_file=True)
        log._setup_logging_from_conf(self.CONF, 'test', 'test')
        handler_mock.assert_called_once_with(path_mock.return_value)
        self.assertEqual(self.log_handlers[0], handler_mock.return_value)

    @mock.patch('logging.handlers.WatchedFileHandler')
    @mock.patch('oslo_log.log._get_log_file_path', return_value='test.conf')
    @mock.patch('platform.system', return_value='Windows')
    def test_watchlog_on_windows(self, platform_mock, path_mock, handler_mock):
        self.config(watch_log_file=True)
        log._setup_logging_from_conf(self.CONF, 'test', 'test')
        handler_mock.assert_called_once_with(path_mock.return_value)
        self.assertEqual(self.log_handlers[0], handler_mock.return_value)

    @mock.patch('logging.handlers.TimedRotatingFileHandler')
    @mock.patch('oslo_log.log._get_log_file_path', return_value='test.conf')
    def test_timed_rotate_log(self, path_mock, handler_mock):
        rotation_type = 'interval'
        when = 'weekday'
        interval = 2
        backup_count = 2
        self.config(log_rotation_type=rotation_type, log_rotate_interval=interval, log_rotate_interval_type=when, max_logfile_count=backup_count)
        log._setup_logging_from_conf(self.CONF, 'test', 'test')
        handler_mock.assert_called_once_with(path_mock.return_value, when='w2', interval=interval, backupCount=backup_count)
        self.assertEqual(self.log_handlers[0], handler_mock.return_value)

    @mock.patch('logging.handlers.RotatingFileHandler')
    @mock.patch('oslo_log.log._get_log_file_path', return_value='test.conf')
    def test_rotate_log(self, path_mock, handler_mock):
        rotation_type = 'size'
        max_logfile_size_mb = 100
        maxBytes = max_logfile_size_mb * units.Mi
        backup_count = 2
        self.config(log_rotation_type=rotation_type, max_logfile_size_mb=max_logfile_size_mb, max_logfile_count=backup_count)
        log._setup_logging_from_conf(self.CONF, 'test', 'test')
        handler_mock.assert_called_once_with(path_mock.return_value, maxBytes=maxBytes, backupCount=backup_count)
        self.assertEqual(self.log_handlers[0], handler_mock.return_value)