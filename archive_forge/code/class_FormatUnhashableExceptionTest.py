import logging
import sys
from unittest import mock
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
from oslotest import base as test_base
from oslo_log import formatters
from oslo_log import log
class FormatUnhashableExceptionTest(test_base.BaseTestCase):

    def setUp(self):
        super(FormatUnhashableExceptionTest, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(cfg.ConfigOpts()))
        self.conf = self.config_fixture.conf
        log.register_options(self.conf)

    def _unhashable_exception_info(self):

        class UnhashableException(Exception):
            __hash__ = None
        try:
            raise UnhashableException()
        except UnhashableException:
            return sys.exc_info()

    def test_error_summary(self):
        exc_info = self._unhashable_exception_info()
        record = logging.LogRecord('test', logging.ERROR, 'test', 0, 'test message', [], exc_info)
        err_summary = formatters._get_error_summary(record)
        self.assertTrue(err_summary)

    def test_json_format_exception(self):
        exc_info = self._unhashable_exception_info()
        formatter = formatters.JSONFormatter()
        tb = ''.join(formatter.formatException(exc_info))
        self.assertTrue(tb)

    def test_fluent_format_exception(self):
        exc_info = self._unhashable_exception_info()
        formatter = formatters.FluentFormatter()
        tb = formatter.formatException(exc_info)
        self.assertTrue(tb)

    def test_context_format_exception_norecord(self):
        exc_info = self._unhashable_exception_info()
        formatter = formatters.ContextFormatter(config=self.conf)
        tb = formatter.formatException(exc_info)
        self.assertTrue(tb)

    def test_context_format_exception(self):
        exc_info = self._unhashable_exception_info()
        formatter = formatters.ContextFormatter(config=self.conf)
        record = logging.LogRecord('test', logging.ERROR, 'test', 0, 'test message', [], exc_info)
        tb = formatter.format(record)
        self.assertTrue(tb)