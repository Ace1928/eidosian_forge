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
class JSONFormatterTestCase(LogTestBase):

    def setUp(self):
        super(JSONFormatterTestCase, self).setUp()
        self.log = log.getLogger('test-json')
        self._add_handler_with_cleanup(self.log, formatter=formatters.JSONFormatter)
        self._set_log_level_with_cleanup(self.log, logging.DEBUG)

    def test_json_w_context_in_extras(self):
        test_msg = 'This is a %(test)s line'
        test_data = {'test': 'log'}
        local_context = _fake_context()
        self.log.debug(test_msg, test_data, key='value', context=local_context)
        self._validate_json_data('test_json_w_context_in_extras', test_msg, test_data, local_context)

    def test_json_w_fetched_global_context(self):
        test_msg = 'This is a %(test)s line'
        test_data = {'test': 'log'}
        local_context = _fake_context()
        self.log.debug(test_msg, test_data, key='value')
        self._validate_json_data('test_json_w_fetched_global_context', test_msg, test_data, local_context)

    def _validate_json_data(self, testname, test_msg, test_data, ctx):
        data = jsonutils.loads(self.stream.getvalue())
        self.assertTrue(data)
        self.assertIn('extra', data)
        self.assertIn('context', data)
        extra = data['extra']
        context = data['context']
        self.assertNotIn('context', extra)
        self.assertEqual('value', extra['key'])
        self.assertEqual(ctx.user, context['user'])
        self.assertEqual(ctx.user_name, context['user_name'])
        self.assertEqual(ctx.project_name, context['project_name'])
        self.assertEqual('test-json', data['name'])
        self.assertIn('request_id', context)
        self.assertEqual(ctx.request_id, context['request_id'])
        self.assertIn('global_request_id', context)
        self.assertEqual(ctx.global_request_id, context['global_request_id'])
        self.assertEqual(test_msg % test_data, data['message'])
        self.assertEqual(test_msg, data['msg'])
        self.assertEqual(test_data, data['args'])
        self.assertEqual('test_log.py', data['filename'])
        self.assertEqual(testname, data['funcname'])
        self.assertEqual('DEBUG', data['levelname'])
        self.assertEqual(logging.DEBUG, data['levelno'])
        self.assertFalse(data['traceback'])

    def test_json_exception(self):
        test_msg = 'This is %s'
        test_data = 'exceptional'
        try:
            raise Exception('This is exceptional')
        except Exception:
            self.log.exception(test_msg, test_data)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertTrue(data)
        self.assertIn('extra', data)
        self.assertEqual('test-json', data['name'])
        self.assertEqual(test_msg % test_data, data['message'])
        self.assertEqual(test_msg, data['msg'])
        self.assertEqual([test_data], data['args'])
        self.assertEqual('ERROR', data['levelname'])
        self.assertEqual(logging.ERROR, data['levelno'])
        self.assertTrue(data['traceback'])

    def test_json_with_extra(self):
        test_msg = 'This is a %(test)s line'
        test_data = {'test': 'log'}
        extra_data = {'special_user': 'user1', 'special_tenant': 'unicorns'}
        self.log.debug(test_msg, test_data, key='value', extra=extra_data)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertTrue(data)
        self.assertIn('extra', data)
        for k, v in extra_data.items():
            self.assertIn(k, data['extra'])
            self.assertEqual(v, data['extra'][k])

    def test_json_with_extra_keys(self):
        test_msg = 'This is a %(test)s line'
        test_data = {'test': 'log'}
        extra_keys = ['special_tenant', 'special_user']
        special_tenant = 'unicorns'
        special_user = 'user2'
        self.log.debug(test_msg, test_data, key='value', extra_keys=extra_keys, special_tenant=special_tenant, special_user=special_user)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertTrue(data)
        self.assertIn('extra', data)
        self.assertIn(extra_keys[0], data['extra'])
        self.assertEqual(special_tenant, data['extra'][extra_keys[0]])
        self.assertIn(extra_keys[1], data['extra'])
        self.assertEqual(special_user, data['extra'][extra_keys[1]])

    def test_can_process_strings(self):
        expected = b'\\u2622'
        expected = '\\\\xe2\\\\x98\\\\xa2'
        self.log.info(b'%s', '☢'.encode('utf8'))
        self.assertIn(expected, self.stream.getvalue())

    def test_exception(self):
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        try:
            raise RuntimeError('test_exception')
        except RuntimeError:
            self.log.warning('testing', context=ctxt)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertIn('error_summary', data)
        self.assertEqual('RuntimeError: test_exception', data['error_summary'])

    def test_no_exception(self):
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        self.log.info('testing', context=ctxt)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertIn('error_summary', data)
        self.assertEqual('', data['error_summary'])

    def test_exception_without_exc_info_passed(self):
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        try:
            raise RuntimeError('test_exception\ntraceback\nfrom\nremote error')
        except RuntimeError:
            self.log.warning('testing', context=ctxt)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertIn('error_summary', data)
        self.assertEqual('RuntimeError: test_exception', data['error_summary'])

    def test_exception_with_exc_info_passed(self):
        ctxt = _fake_context()
        ctxt.request_id = str('99')
        try:
            raise RuntimeError('test_exception\ntraceback\nfrom\nremote error')
        except RuntimeError:
            self.log.exception('testing', context=ctxt)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertIn('error_summary', data)
        self.assertEqual('RuntimeError: test_exception\ntraceback\nfrom\nremote error', data['error_summary'])

    def test_fallback(self):

        class MyObject(object):

            def __str__(self):
                return 'str'

            def __repr__(self):
                return 'repr'
        obj = MyObject()
        self.log.debug('obj=%s', obj)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertEqual('obj=str', data['message'])
        self.assertEqual(['repr'], data['args'])

    def test_extra_args_filtered(self):
        test_msg = 'This is a %(test)s line %%(unused)'
        test_data = {'test': 'log', 'unused': 'removeme'}
        self.log.debug(test_msg, test_data)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertNotIn('unused', data['args'])

    def test_entire_dict(self):
        test_msg = 'This is a %s dict'
        test_data = {'test': 'log', 'other': 'value'}
        self.log.debug(test_msg, test_data)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertEqual(test_data, data['args'])