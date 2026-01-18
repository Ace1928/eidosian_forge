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
class FancyRecordTestCase(LogTestBase):
    """Test how we handle fancy record keys that are not in the
    base python logging.
    """

    def setUp(self):
        super(FancyRecordTestCase, self).setUp()
        self.config(logging_context_format_string='%(color)s [%(request_id)s]: %(instance)s%(resource)s%(message)s', logging_default_format_string='%(missing)s: %(message)s')
        self.colorlog = log.getLogger()
        self._add_handler_with_cleanup(self.colorlog, handlers.ColorHandler)
        self._set_log_level_with_cleanup(self.colorlog, logging.DEBUG)

    def test_unsupported_key_in_log_msg(self):
        error = sys.stderr
        sys.stderr = io.StringIO()
        self.colorlog.info('foo')
        self.assertNotEqual(-1, sys.stderr.getvalue().find("KeyError: 'missing'"))
        sys.stderr = error

    def _validate_keys(self, ctxt, keyed_log_string):
        infocolor = handlers.ColorHandler.LEVEL_COLORS[logging.INFO]
        warncolor = handlers.ColorHandler.LEVEL_COLORS[logging.WARN]
        info_msg = 'info'
        warn_msg = 'warn'
        infoexpected = '%s %s %s' % (infocolor, keyed_log_string, info_msg)
        warnexpected = '%s %s %s' % (warncolor, keyed_log_string, warn_msg)
        self.colorlog.info(info_msg, context=ctxt)
        self.assertIn(infoexpected, self.stream.getvalue())
        self.assertEqual('\x1b[00;36m', infocolor)
        self.colorlog.warn(warn_msg, context=ctxt)
        self.assertIn(infoexpected, self.stream.getvalue())
        self.assertIn(warnexpected, self.stream.getvalue())
        self.assertEqual('\x1b[01;33m', warncolor)

    def test_fancy_key_in_log_msg(self):
        ctxt = _fake_context()
        self._validate_keys(ctxt, '[%s]:' % ctxt.request_id)

    def test_instance_key_in_log_msg(self):
        ctxt = _fake_context()
        ctxt.resource_uuid = '1234'
        self._validate_keys(ctxt, '[%s]: [instance: %s]' % (ctxt.request_id, ctxt.resource_uuid))

    def test_resource_key_in_log_msg(self):
        color = handlers.ColorHandler.LEVEL_COLORS[logging.INFO]
        ctxt = _fake_context()
        resource = 'resource-202260f9-1224-490d-afaf-6a744c13141f'
        fake_resource = {'name': resource}
        message = 'info'
        self.colorlog.info(message, context=ctxt, resource=fake_resource)
        expected = '%s [%s]: [%s] %s\x1b[00m\n' % (color, ctxt.request_id, resource, message)
        self.assertEqual(expected, self.stream.getvalue())

    def test_resource_key_dict_in_log_msg(self):
        color = handlers.ColorHandler.LEVEL_COLORS[logging.INFO]
        ctxt = _fake_context()
        type = 'fake_resource'
        resource_id = '202260f9-1224-490d-afaf-6a744c13141f'
        fake_resource = {'type': type, 'id': resource_id}
        message = 'info'
        self.colorlog.info(message, context=ctxt, resource=fake_resource)
        expected = '%s [%s]: [%s-%s] %s\x1b[00m\n' % (color, ctxt.request_id, type, resource_id, message)
        self.assertEqual(expected, self.stream.getvalue())