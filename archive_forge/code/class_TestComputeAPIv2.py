from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
class TestComputeAPIv2(utils.TestCase):

    def setUp(self):
        super(TestComputeAPIv2, self).setUp()
        sess = session.Session()
        self.api = compute.APIv2(session=sess, endpoint=FAKE_URL)
        self.requests_mock = self.useFixture(fixture.Fixture())