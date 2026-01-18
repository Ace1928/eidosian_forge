from keystoneauth1 import session
from requests_mock.contrib import fixture
from openstackclient.api import image_v1
from openstackclient.tests.unit import utils
class TestImageAPIv1(utils.TestCase):

    def setUp(self):
        super(TestImageAPIv1, self).setUp()
        sess = session.Session()
        self.api = image_v1.APIv1(session=sess, endpoint=FAKE_URL)
        self.requests_mock = self.useFixture(fixture.Fixture())