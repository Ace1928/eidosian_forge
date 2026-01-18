import uuid
import fixtures
from keystoneauth1 import fixture as ks_fixture
from keystoneauth1.tests.unit import utils as test_utils
def _authenticate_server(self, response):
    self.called_auth_server = True
    return response.headers.get('www-authenticate') == self.pass_header