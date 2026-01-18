from unittest import mock
from oslo_serialization import jsonutils
import testtools
from heatclient.tests.unit import fakes
from heatclient.v1 import build_info
class BuildInfoManagerTest(testtools.TestCase):

    def setUp(self):
        super(BuildInfoManagerTest, self).setUp()
        self.client = mock.Mock()
        self.client.get.return_value = fakes.FakeHTTPResponse(200, None, {'content-type': 'application/json'}, jsonutils.dumps('body'))
        self.manager = build_info.BuildInfoManager(self.client)

    def test_build_info_makes_a_call_to_the_api(self):
        self.manager.build_info()
        self.client.get.assert_called_once_with('/build_info')

    def test_build_info_returns_the_response_body(self):
        response = self.manager.build_info()
        self.assertEqual('body', response)