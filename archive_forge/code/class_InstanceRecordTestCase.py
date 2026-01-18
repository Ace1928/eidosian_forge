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
class InstanceRecordTestCase(LogTestBase):

    def setUp(self):
        super(InstanceRecordTestCase, self).setUp()
        self.config(logging_context_format_string='[%(request_id)s]: %(instance)s%(resource)s%(message)s', logging_default_format_string='%(instance)s%(resource)s%(message)s')
        self.log = log.getLogger()
        self._add_handler_with_cleanup(self.log)
        self._set_log_level_with_cleanup(self.log, logging.DEBUG)

    def test_instance_dict_in_context_log_msg(self):
        ctxt = _fake_context()
        uuid = 'C9B7CCC6-8A12-4C53-A736-D7A1C36A62F3'
        fake_resource = {'uuid': uuid}
        message = 'info'
        self.log.info(message, context=ctxt, instance=fake_resource)
        expected = '[instance: %s]' % uuid
        self.assertIn(expected, self.stream.getvalue())

    def test_instance_dict_in_default_log_msg(self):
        uuid = 'C9B7CCC6-8A12-4C53-A736-D7A1C36A62F3'
        fake_resource = {'uuid': uuid}
        message = 'info'
        self.log.info(message, instance=fake_resource)
        expected = '[instance: %s]' % uuid
        self.assertIn(expected, self.stream.getvalue())

    def test_instance_uuid_as_arg_in_context_log_msg(self):
        ctxt = _fake_context()
        uuid = 'C9B7CCC6-8A12-4C53-A736-D7A1C36A62F3'
        message = 'info'
        self.log.info(message, context=ctxt, instance_uuid=uuid)
        expected = '[instance: %s]' % uuid
        self.assertIn(expected, self.stream.getvalue())

    def test_instance_uuid_as_arg_in_default_log_msg(self):
        uuid = 'C9B7CCC6-8A12-4C53-A736-D7A1C36A62F3'
        message = 'info'
        self.log.info(message, instance_uuid=uuid)
        expected = '[instance: %s]' % uuid
        self.assertIn(expected, self.stream.getvalue())

    def test_instance_uuid_from_context_in_context_log_msg(self):
        ctxt = _fake_context()
        ctxt.instance_uuid = 'CCCCCCCC-8A12-4C53-A736-D7A1C36A62F3'
        message = 'info'
        self.log.info(message, context=ctxt)
        expected = '[instance: %s]' % ctxt.instance_uuid
        self.assertIn(expected, self.stream.getvalue())

    def test_resource_uuid_from_context_in_context_log_msg(self):
        ctxt = _fake_context()
        ctxt.resource_uuid = 'RRRRRRRR-8A12-4C53-A736-D7A1C36A62F3'
        message = 'info'
        self.log.info(message, context=ctxt)
        expected = '[instance: %s]' % ctxt.resource_uuid
        self.assertIn(expected, self.stream.getvalue())

    def test_instance_from_context_in_context_log_msg(self):
        ctxt = _fake_context()
        ctxt.instance = 'IIIIIIII-8A12-4C53-A736-D7A1C36A62F3'
        message = 'info'
        self.log.info(message, context=ctxt)
        expected = '[instance: %s]' % ctxt.instance
        self.assertIn(expected, self.stream.getvalue())