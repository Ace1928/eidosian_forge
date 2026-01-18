from keystoneauth1 import session
from openstackclient.api import object_store_v1 as object_store
from openstackclient.tests.unit import utils
class TestObjectv1(utils.TestCommand):

    def setUp(self):
        super(TestObjectv1, self).setUp()
        self.app.client_manager.session = session.Session()
        self.app.client_manager.object_store = object_store.APIv1(session=self.app.client_manager.session, endpoint=ENDPOINT)