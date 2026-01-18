import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
class DiscoveryFailures(utils.TestCase):
    TEST_ROOT_URL = 'http://127.0.0.1:5000/'

    def test_connection_error(self):
        self.requests_mock.get(self.TEST_ROOT_URL, exc=exceptions.ConnectionError)
        sess = session.Session()
        p = identity.generic.password.Password(self.TEST_ROOT_URL)
        self.assertRaises(exceptions.DiscoveryFailure, p.get_auth_ref, sess)

    def test_client_exception(self):
        self.requests_mock.get(self.TEST_ROOT_URL, exc=exceptions.ClientException)
        sess = session.Session()
        p = identity.generic.password.Password(self.TEST_ROOT_URL)
        self.assertRaises(exceptions.ClientException, p.get_auth_ref, sess)

    def test_ssl_error(self):
        self.requests_mock.get(self.TEST_ROOT_URL, exc=exceptions.SSLError)
        sess = session.Session()
        p = identity.generic.password.Password(self.TEST_ROOT_URL)
        self.assertRaises(exceptions.DiscoveryFailure, p.get_auth_ref, sess)