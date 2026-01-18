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
class LogTestBase(BaseTestCase):
    """Base test class that provides some convenience functions."""

    def _add_handler_with_cleanup(self, log_instance, handler=None, formatter=None):
        """Add a log handler to a log instance.

        This function should be used to add handlers to loggers in test cases
        instead of directly adding them to ensure that the handler is
        correctly removed at the end of the test.  Otherwise the handler may
        be left on the logger and interfere with subsequent tests.

        :param log_instance: The log instance to which the handler will be
            added.
        :param handler: The handler class to be added.  Must be the class
            itself, not an instance.
        :param formatter: The formatter class to set on the handler.  Must be
            the class itself, not an instance.
        """
        self.stream = io.StringIO()
        if handler is None:
            handler = logging.StreamHandler
        self.handler = handler(self.stream)
        if formatter is None:
            formatter = formatters.ContextFormatter
        self.handler.setFormatter(formatter())
        log_instance.logger.addHandler(self.handler)
        self.addCleanup(log_instance.logger.removeHandler, self.handler)

    def _set_log_level_with_cleanup(self, log_instance, level):
        """Set the log level of a logger for the duration of a test.

        Use this function to set the log level of a logger and add the
        necessary cleanup to reset it back to default at the end of the test.

        :param log_instance: The logger whose level will be changed.
        :param level: The new log level to use.
        """
        self.level = log_instance.logger.getEffectiveLevel()
        log_instance.logger.setLevel(level)
        self.addCleanup(log_instance.logger.setLevel, self.level)