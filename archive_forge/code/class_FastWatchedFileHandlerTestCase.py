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
@testtools.skipIf(platform.system() != 'Linux', 'pyinotify library works on Linux platform only.')
class FastWatchedFileHandlerTestCase(BaseTestCase):

    def setUp(self):
        super(FastWatchedFileHandlerTestCase, self).setUp()

    def _config(self):
        os_level, log_path = tempfile.mkstemp()
        log_dir_path = os.path.dirname(log_path)
        log_file_path = os.path.basename(log_path)
        self.CONF(['--log-dir', log_dir_path, '--log-file', log_file_path])
        self.config(use_stderr=False)
        self.config(watch_log_file=True)
        log.setup(self.CONF, 'test', 'test')
        return log_path

    def test_instantiate(self):
        self._config()
        logger = log._loggers[None].logger
        self.assertEqual(1, len(logger.handlers))
        from oslo_log import watchers
        self.assertIsInstance(logger.handlers[0], watchers.FastWatchedFileHandler)

    def test_log(self):
        log_path = self._config()
        logger = log._loggers[None].logger
        text = 'Hello World!'
        logger.info(text)
        with open(log_path, 'r') as f:
            file_content = f.read()
        self.assertIn(text, file_content)

    def test_move(self):
        log_path = self._config()
        os_level_dst, log_path_dst = tempfile.mkstemp()
        os.rename(log_path, log_path_dst)
        time.sleep(6)
        self.assertTrue(os.path.exists(log_path))

    def test_remove(self):
        log_path = self._config()
        os.remove(log_path)
        time.sleep(6)
        self.assertTrue(os.path.exists(log_path))