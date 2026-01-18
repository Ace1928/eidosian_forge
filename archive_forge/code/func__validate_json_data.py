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