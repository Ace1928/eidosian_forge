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
class IsDebugEnabledTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(IsDebugEnabledTestCase, self).setUp()
        self.config_fixture = self.useFixture(fixture_config.Config(cfg.ConfigOpts()))
        self.config = self.config_fixture.config
        self.CONF = self.config_fixture.conf
        log.register_options(self.config_fixture.conf)

    def _test_is_debug_enabled(self, debug=False):
        self.config(debug=debug)
        self.assertEqual(debug, log.is_debug_enabled(self.CONF))

    def test_is_debug_enabled_off(self):
        self._test_is_debug_enabled()

    def test_is_debug_enabled_on(self):
        self._test_is_debug_enabled(debug=True)