import uuid
from openstackclient.tests.functional.network.v2 import common
class NetworkQosTests(common.NetworkTests):

    def setUp(self):
        super().setUp()
        if not self.is_extension_enabled('qos'):
            self.skipTest('No qos extension present')