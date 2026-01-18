import fixtures
import testresources
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3.contrib import simple_cert
def _mock_request_method(self, method=None, body=None):
    return self.useFixture(fixtures.MockPatchObject(self.client, method, autospec=True, return_value=(self.resp, body))).mock