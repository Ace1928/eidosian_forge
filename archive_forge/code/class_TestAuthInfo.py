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
class TestAuthInfo(common_auth.AuthTestMixin, testcase.TestCase):

    def setUp(self):
        super(TestAuthInfo, self).setUp()
        auth.core.load_auth_methods()

    def test_unsupported_auth_method(self):
        auth_data = {'methods': ['abc']}
        auth_data['abc'] = {'test': 'test'}
        auth_data = {'identity': auth_data}
        self.assertRaises(exception.AuthMethodNotSupported, auth.core.AuthInfo.create, auth_data)

    def test_missing_auth_method_data(self):
        auth_data = {'methods': ['password']}
        auth_data = {'identity': auth_data}
        self.assertRaises(exception.ValidationError, auth.core.AuthInfo.create, auth_data)

    def test_project_name_no_domain(self):
        auth_data = self.build_authentication_request(username='test', password='test', project_name='abc')['auth']
        self.assertRaises(exception.ValidationError, auth.core.AuthInfo.create, auth_data)

    def test_both_project_and_domain_in_scope(self):
        auth_data = self.build_authentication_request(user_id='test', password='test', project_name='test', domain_name='test')['auth']
        self.assertRaises(exception.ValidationError, auth.core.AuthInfo.create, auth_data)

    def test_get_method_names_duplicates(self):
        auth_data = self.build_authentication_request(token='test', user_id='test', password='test')['auth']
        auth_data['identity']['methods'] = ['password', 'token', 'password', 'password']
        auth_info = auth.core.AuthInfo.create(auth_data)
        self.assertEqual(['password', 'token'], auth_info.get_method_names())

    def test_get_method_data_invalid_method(self):
        auth_data = self.build_authentication_request(user_id='test', password='test')['auth']
        auth_info = auth.core.AuthInfo.create(auth_data)
        method_name = uuid.uuid4().hex
        self.assertRaises(exception.ValidationError, auth_info.get_method_data, method_name)