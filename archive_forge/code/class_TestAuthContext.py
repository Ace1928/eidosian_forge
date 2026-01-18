import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
class TestAuthContext(unit.TestCase):

    def setUp(self):
        super(TestAuthContext, self).setUp()
        self.auth_context = auth.core.AuthContext()

    def test_pick_lowest_expires_at(self):
        expires_at_1 = utils.isotime(timeutils.utcnow())
        expires_at_2 = utils.isotime(timeutils.utcnow() + datetime.timedelta(seconds=10))
        self.auth_context['expires_at'] = expires_at_1
        self.auth_context['expires_at'] = expires_at_2
        self.assertEqual(expires_at_1, self.auth_context['expires_at'])

    def test_identity_attribute_conflict(self):
        for identity_attr in auth.core.AuthContext.IDENTITY_ATTRIBUTES:
            self.auth_context[identity_attr] = uuid.uuid4().hex
            if identity_attr == 'expires_at':
                continue
            self.assertRaises(exception.Unauthorized, operator.setitem, self.auth_context, identity_attr, uuid.uuid4().hex)

    def test_identity_attribute_conflict_with_none_value(self):
        for identity_attr in auth.core.AuthContext.IDENTITY_ATTRIBUTES:
            self.auth_context[identity_attr] = None
            if identity_attr == 'expires_at':
                self.auth_context['expires_at'] = uuid.uuid4().hex
                continue
            self.assertRaises(exception.Unauthorized, operator.setitem, self.auth_context, identity_attr, uuid.uuid4().hex)

    def test_non_identity_attribute_conflict_override(self):
        attr_name = uuid.uuid4().hex
        attr_val_1 = uuid.uuid4().hex
        attr_val_2 = uuid.uuid4().hex
        self.auth_context[attr_name] = attr_val_1
        self.auth_context[attr_name] = attr_val_2
        self.assertEqual(attr_val_2, self.auth_context[attr_name])