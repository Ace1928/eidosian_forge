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
class ContextFormatterTestCase(LogTestBase):

    def setUp(self):
        super(ContextFormatterTestCase, self).setUp()
        self.config(logging_context_format_string='HAS CONTEXT [%(request_id)s]: %(message)s', logging_default_format_string='NOCTXT: %(message)s', logging_debug_format_suffix='--DBG')
        self.log = log.getLogger('')
        self._add_handler_with_cleanup(self.log)
        self._set_log_level_with_cleanup(self.log, logging.DEBUG)
        self.trans_fixture = self.useFixture(fixture_trans.Translation())

    def test_uncontextualized_log(self):
        message = 'foo'
        self.log.info(message)
        self.assertEqual('NOCTXT: %s\n' % message, self.stream.getvalue())

    def test_contextualized_log(self):
        ctxt = _fake_context()
        message = 'bar'
        self.log.info(message, context=ctxt)
        expected = 'HAS CONTEXT [%s]: %s\n' % (ctxt.request_id, message)
        self.assertEqual(expected, self.stream.getvalue())

    def test_context_is_taken_from_tls_variable(self):
        ctxt = _fake_context()
        message = 'bar'
        self.log.info(message)
        expected = 'HAS CONTEXT [%s]: %s\n' % (ctxt.request_id, message)
        self.assertEqual(expected, self.stream.getvalue())

    def test_contextual_information_is_imparted_to_3rd_party_log_records(self):
        ctxt = _fake_context()
        sa_log = logging.getLogger('sqlalchemy.engine')
        sa_log.setLevel(logging.INFO)
        message = 'emulate logging within sqlalchemy'
        sa_log.info(message)
        expected = 'HAS CONTEXT [%s]: %s\n' % (ctxt.request_id, message)
        self.assertEqual(expected, self.stream.getvalue())

    def test_message_logging_3rd_party_log_records(self):
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        sa_log = logging.getLogger('sqlalchemy.engine')
        sa_log.setLevel(logging.INFO)
        message = self.trans_fixture.lazy('test ' + chr(128))
        sa_log.info(message)
        expected = 'HAS CONTEXT [%s]: %s\n' % (ctxt.request_id, str(message))
        self.assertEqual(expected, self.stream.getvalue())

    def test_debugging_log(self):
        message = 'baz'
        self.log.debug(message)
        self.assertEqual('NOCTXT: %s --DBG\n' % message, self.stream.getvalue())

    def test_message_logging(self):
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        message = self.trans_fixture.lazy('test ' + chr(128))
        self.log.info(message, context=ctxt)
        expected = 'HAS CONTEXT [%s]: %s\n' % (ctxt.request_id, str(message))
        self.assertEqual(expected, self.stream.getvalue())

    def test_exception_logging(self):
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        message = self.trans_fixture.lazy('test ' + chr(128))
        try:
            raise RuntimeError('test_exception_logging')
        except RuntimeError:
            self.log.warning(message, context=ctxt)
        expected = 'RuntimeError: test_exception_logging\n'
        self.assertTrue(self.stream.getvalue().endswith(expected))

    def test_skip_logging_builtin_exceptions(self):
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        message = self.trans_fixture.lazy('test ' + chr(128))
        ignored_exceptions = [ValueError, TypeError, KeyError, AttributeError, ImportError]
        for ignore in ignored_exceptions:
            try:
                raise ignore('test_exception_logging')
            except ignore as e:
                self.log.warning(message, context=ctxt)
                expected = '{}: {}'.format(e.__class__.__name__, e)
            self.assertNotIn(expected, self.stream.getvalue())

    def test_exception_logging_format_string(self):
        self.config(logging_context_format_string='A %(error_summary)s B')
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        message = self.trans_fixture.lazy('test ' + chr(128))
        try:
            raise RuntimeError('test_exception_logging')
        except RuntimeError:
            self.log.warning(message, context=ctxt)
        expected = 'A RuntimeError: test_exception_logging'
        self.assertTrue(self.stream.getvalue().startswith(expected))

    def test_no_exception_logging_format_string(self):
        self.config(logging_context_format_string='%(error_summary)s')
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        message = self.trans_fixture.lazy('test ' + chr(128))
        self.log.info(message, context=ctxt)
        expected = '-\n'
        self.assertTrue(self.stream.getvalue().startswith(expected))

    def test_unicode_conversion_in_adapter(self):
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        message = 'Exception is (%s)'
        ex = Exception(self.trans_fixture.lazy('test' + chr(128)))
        self.log.debug(message, ex, context=ctxt)
        message = str(message) % ex
        expected = 'HAS CONTEXT [%s]: %s --DBG\n' % (ctxt.request_id, message)
        self.assertEqual(expected, self.stream.getvalue())

    def test_unicode_conversion_in_formatter(self):
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        no_adapt_log = logging.getLogger('no_adapt')
        no_adapt_log.setLevel(logging.INFO)
        message = 'Exception is (%s)'
        ex = Exception(self.trans_fixture.lazy('test' + chr(128)))
        no_adapt_log.info(message, ex)
        message = str(message) % ex
        expected = 'HAS CONTEXT [%s]: %s\n' % (ctxt.request_id, message)
        self.assertEqual(expected, self.stream.getvalue())

    def test_user_identity_logging(self):
        self.config(logging_context_format_string='HAS CONTEXT [%(request_id)s %(user_identity)s]: %(message)s')
        ctxt = _fake_context()
        ctxt.request_id = '99'
        message = 'test'
        self.log.info(message, context=ctxt)
        expected = 'HAS CONTEXT [%s %s %s %s %s %s %s]: %s\n' % (ctxt.request_id, ctxt.user, ctxt.project_id, ctxt.domain, ctxt.system_scope, ctxt.user_domain, ctxt.project_domain, str(message))
        self.assertEqual(expected, self.stream.getvalue())

    def test_global_request_id_logging(self):
        fmt_str = 'HAS CONTEXT [%(request_id)s %(global_request_id)s]: %(message)s'
        self.config(logging_context_format_string=fmt_str)
        ctxt = _fake_context()
        ctxt.request_id = '99'
        message = 'test'
        self.log.info(message, context=ctxt)
        expected = 'HAS CONTEXT [%s %s]: %s\n' % (ctxt.request_id, ctxt.global_request_id, str(message))
        self.assertEqual(expected, self.stream.getvalue())

    def test_user_identity_logging_set_format(self):
        self.config(logging_context_format_string='HAS CONTEXT [%(request_id)s %(user_identity)s]: %(message)s', logging_user_identity_format='%(user)s %(project)s')
        ctxt = _fake_context()
        ctxt.request_id = '99'
        message = 'test'
        self.log.info(message, context=ctxt)
        expected = 'HAS CONTEXT [%s %s %s]: %s\n' % (ctxt.request_id, ctxt.user, ctxt.project_id, str(message))
        self.assertEqual(expected, self.stream.getvalue())

    @mock.patch('datetime.datetime', get_fake_datetime(datetime.datetime(2015, 12, 16, 13, 54, 26, 517893)))
    @mock.patch('dateutil.tz.tzlocal', new=mock.Mock(return_value=tz.tzutc()))
    def test_rfc5424_isotime_format(self):
        self.config(logging_default_format_string='%(isotime)s %(message)s')
        message = 'test'
        expected = '2015-12-16T13:54:26.517893+00:00 %s\n' % message
        self.log.info(message)
        self.assertEqual(expected, self.stream.getvalue())

    @mock.patch('datetime.datetime', get_fake_datetime(datetime.datetime(2015, 12, 16, 13, 54, 26)))
    @mock.patch('time.time', new=mock.Mock(return_value=1450274066.0))
    @mock.patch('dateutil.tz.tzlocal', new=mock.Mock(return_value=tz.tzutc()))
    def test_rfc5424_isotime_format_no_microseconds(self):
        self.config(logging_default_format_string='%(isotime)s %(message)s')
        message = 'test'
        expected = '2015-12-16T13:54:26.000000+00:00 %s\n' % message
        self.log.info(message)
        self.assertEqual(expected, self.stream.getvalue())

    def test_can_process_strings(self):
        expected = b'\xe2\x98\xa2'
        expected = '\\xe2\\x98\\xa2'
        self.log.info(b'%s', '☢'.encode('utf8'))
        self.assertIn(expected, self.stream.getvalue())

    def test_dict_args_with_unicode(self):
        msg = '%(thing)s'
        arg = {'thing': 'Æ\x91Æ¡Æ¡'}
        self.log.info(msg, arg)
        self.assertIn(arg['thing'], self.stream.getvalue())