import keystoneclient
from keystoneclient.tests.functional import base
class V3ClientVersionTestCase(base.V3ClientTestCase):

    def test_version(self):
        self.assertIsInstance(self.client, keystoneclient.v3.client.Client)