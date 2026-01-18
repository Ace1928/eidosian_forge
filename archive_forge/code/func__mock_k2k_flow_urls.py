import copy
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from keystoneauth1.tests.unit import utils
def _mock_k2k_flow_urls(self, redirect_code=302):
    self.requests_mock.get(self.TEST_URL, json={'version': fixture.V3Discovery(self.TEST_URL)}, headers={'Content-Type': 'application/json'})
    self.requests_mock.register_uri('POST', self.REQUEST_ECP_URL, content=bytes(k2k_fixtures.ECP_ENVELOPE, 'latin-1'), headers={'Content-Type': 'application/vnd.paos+xml'}, status_code=200)
    self.requests_mock.register_uri('POST', self.SP_URL, content=bytes(k2k_fixtures.TOKEN_BASED_ECP, 'latin-1'), headers={'Content-Type': 'application/vnd.paos+xml'}, status_code=redirect_code)
    self.requests_mock.register_uri('GET', self.SP_AUTH_URL, json=k2k_fixtures.UNSCOPED_TOKEN, headers={'X-Subject-Token': k2k_fixtures.UNSCOPED_TOKEN_HEADER})