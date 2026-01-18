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
def _validate_keys(self, ctxt, keyed_log_string):
    info_message = 'info'
    infoexpected = '%s %s\n' % (keyed_log_string, info_message)
    warn_message = 'warn'
    warnexpected = '%s %s\n' % (keyed_log_string, warn_message)
    self.mylog.info(info_message, context=ctxt)
    self.assertEqual(infoexpected, self.stream.getvalue())
    self.mylog.warn(warn_message, context=ctxt)
    self.assertEqual(infoexpected + warnexpected, self.stream.getvalue())