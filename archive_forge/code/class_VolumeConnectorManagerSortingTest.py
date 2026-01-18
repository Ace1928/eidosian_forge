import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
class VolumeConnectorManagerSortingTest(VolumeConnectorManagerTestBase):

    def setUp(self):
        super(VolumeConnectorManagerSortingTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses_sorting)
        self.mgr = ironicclient.v1.volume_connector.VolumeConnectorManager(self.api)

    def test_volume_connectors_list_sort_key(self):
        volume_connectors = self.mgr.list(sort_key='updated_at')
        expect = [('GET', '/v1/volume/connectors/?sort_key=updated_at', {}, None)]
        expect_connectors = [CONNECTOR2, CONNECTOR1]
        self._validate_list(expect, expect_connectors, volume_connectors)

    def test_volume_connectors_list_sort_dir(self):
        volume_connectors = self.mgr.list(sort_dir='desc')
        expect = [('GET', '/v1/volume/connectors/?sort_dir=desc', {}, None)]
        expect_connectors = [CONNECTOR2, CONNECTOR1]
        self._validate_list(expect, expect_connectors, volume_connectors)