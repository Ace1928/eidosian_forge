import datetime
import os
import time
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
import oslo_cache
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import pbr.version
import testresources
from testtools import matchers
import webob
import webob.dec
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import client_fixtures
class TimezoneFixture(fixtures.Fixture):

    @staticmethod
    def supported():
        return hasattr(time, 'tzset')

    def __init__(self, new_tz):
        super(TimezoneFixture, self).__init__()
        self.tz = new_tz
        self.old_tz = os.environ.get('TZ')

    def setUp(self):
        super(TimezoneFixture, self).setUp()
        if not self.supported():
            raise NotImplementedError('timezone override is not supported.')
        os.environ['TZ'] = self.tz
        time.tzset()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if self.old_tz is not None:
            os.environ['TZ'] = self.old_tz
        elif 'TZ' in os.environ:
            del os.environ['TZ']
        time.tzset()