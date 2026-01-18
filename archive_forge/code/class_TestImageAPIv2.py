from keystoneauth1 import session
from requests_mock.contrib import fixture
from openstackclient.api import image_v2
from openstackclient.tests.unit import utils
class TestImageAPIv2(utils.TestCase):

    def setUp(self):
        super(TestImageAPIv2, self).setUp()
        sess = session.Session()
        self.api = image_v2.APIv2(session=sess, endpoint=FAKE_URL)
        self.requests_mock = self.useFixture(fixture.Fixture())