import fixtures
import http.client as http_client
import logging
import testresources
import uuid
import webob.dec
from oslo_config import cfg
from keystoneauth1 import exceptions as ksa_exceptions
from keystonemiddleware import oauth2_token
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware\
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit import client_fixtures
from keystonemiddleware.tests.unit import utils
class FilterFactoryTest(utils.BaseTestCase):

    def test_filter_factory(self):
        conf = {}
        auth_filter = oauth2_token.filter_factory(conf)
        m = auth_filter(FakeOauth2TokenV3App())
        self.assertIsInstance(m, oauth2_token.OAuth2Protocol)