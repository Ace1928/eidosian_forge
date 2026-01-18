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
class TimeFixture(fixtures.Fixture):

    def __init__(self, new_time, normalize=True):
        super(TimeFixture, self).__init__()
        if isinstance(new_time, str):
            new_time = timeutils.parse_isotime(new_time)
        if normalize:
            new_time = timeutils.normalize_time(new_time)
        self.new_time = new_time

    def setUp(self):
        super(TimeFixture, self).setUp()
        timeutils.set_time_override(self.new_time)
        self.addCleanup(timeutils.clear_time_override)