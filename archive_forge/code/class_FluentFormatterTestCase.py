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
class FluentFormatterTestCase(LogTestBase):

    def setUp(self):
        super(FluentFormatterTestCase, self).setUp()
        self.log = log.getLogger('test-fluent')
        self._add_handler_with_cleanup(self.log, handler=DictStreamHandler, formatter=formatters.FluentFormatter)
        self._set_log_level_with_cleanup(self.log, logging.DEBUG)

    def test_fluent(self):
        test_msg = 'This is a %(test)s line'
        test_data = {'test': 'log'}
        local_context = _fake_context()
        self.log.debug(test_msg, test_data, key='value', context=local_context)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertIn('lineno', data)
        self.assertIn('extra', data)
        extra = data['extra']
        context = data['context']
        self.assertEqual('value', extra['key'])
        self.assertEqual(local_context.user, context['user'])
        self.assertEqual('test-fluent', data['name'])
        self.assertIn('request_id', context)
        self.assertEqual(local_context.request_id, context['request_id'])
        self.assertIn('global_request_id', context)
        self.assertEqual(local_context.global_request_id, context['global_request_id'])
        self.assertEqual(test_msg % test_data, data['message'])
        self.assertEqual('test_log.py', data['filename'])
        self.assertEqual('test_fluent', data['funcname'])
        self.assertEqual('DEBUG', data['level'])
        self.assertFalse(data['traceback'])

    def test_exception(self):
        local_context = _fake_context()
        try:
            raise RuntimeError('test_exception')
        except RuntimeError:
            self.log.warning('testing', context=local_context)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertIn('error_summary', data)
        self.assertEqual('RuntimeError: test_exception', data['error_summary'])

    def test_no_exception(self):
        local_context = _fake_context()
        self.log.info('testing', context=local_context)
        data = jsonutils.loads(self.stream.getvalue())
        self.assertIn('error_summary', data)
        self.assertEqual('', data['error_summary'])

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
        self.assertEqual('test-fluent', data['name'])
        self.assertEqual(test_msg % test_data, data['message'])
        self.assertEqual('ERROR', data['level'])
        self.assertTrue(data['traceback'])