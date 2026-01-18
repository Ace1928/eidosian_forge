import uuid
import fixtures
from keystoneauth1 import fixture as ks_fixture
from keystoneauth1.tests.unit import utils as test_utils
def _generate_request_header(self, *args, **kwargs):
    return self.challenge_header