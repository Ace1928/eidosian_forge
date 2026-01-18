from unittest import mock
from keystoneauth1 import session
from requests_mock.contrib import fixture
from openstackclient.api import object_store_v1 as object_store
from openstackclient.tests.unit import utils
class TestObjectAPIv1(utils.TestCase):

    def setUp(self):
        super(TestObjectAPIv1, self).setUp()
        sess = session.Session()
        self.api = object_store.APIv1(session=sess, endpoint=FAKE_URL)
        self.requests_mock = self.useFixture(fixture.Fixture())