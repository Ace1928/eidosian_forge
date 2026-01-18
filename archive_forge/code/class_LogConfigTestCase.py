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
class LogConfigTestCase(BaseTestCase):

    def setUp(self):
        super(LogConfigTestCase, self).setUp()
        names = self.create_tempfiles([('logging', MIN_LOG_INI)])
        self.log_config_append = names[0]
        if hasattr(log._load_log_config, 'old_time'):
            del log._load_log_config.old_time

    def test_log_config_append_ok(self):
        self.config(log_config_append=self.log_config_append)
        log.setup(self.CONF, 'test_log_config_append')

    def test_log_config_append_not_exist(self):
        os.remove(self.log_config_append)
        self.config(log_config_append=self.log_config_append)
        self.assertRaises(log.LogConfigError, log.setup, self.CONF, 'test_log_config_append')

    def test_log_config_append_invalid(self):
        names = self.create_tempfiles([('logging', 'squawk')])
        self.log_config_append = names[0]
        self.config(log_config_append=self.log_config_append)
        self.assertRaises(log.LogConfigError, log.setup, self.CONF, 'test_log_config_append')

    def test_log_config_append_unreadable(self):
        os.chmod(self.log_config_append, 0)
        self.config(log_config_append=self.log_config_append)
        self.assertRaises(log.LogConfigError, log.setup, self.CONF, 'test_log_config_append')

    def test_log_config_append_disable_existing_loggers(self):
        self.config(log_config_append=self.log_config_append)
        with mock.patch('logging.config.fileConfig') as fileConfig:
            log.setup(self.CONF, 'test_log_config_append')
        fileConfig.assert_called_once_with(self.log_config_append, disable_existing_loggers=False)