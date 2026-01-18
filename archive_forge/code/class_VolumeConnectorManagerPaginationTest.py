import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
class VolumeConnectorManagerPaginationTest(VolumeConnectorManagerTestBase):

    def setUp(self):
        super(VolumeConnectorManagerPaginationTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = ironicclient.v1.volume_connector.VolumeConnectorManager(self.api)

    def test_volume_connectors_list_limit(self):
        volume_connectors = self.mgr.list(limit=1)
        expect = [('GET', '/v1/volume/connectors/?limit=1', {}, None)]
        expect_connectors = [CONNECTOR1]
        self._validate_list(expect, expect_connectors, volume_connectors)

    def test_volume_connectors_list_marker(self):
        volume_connectors = self.mgr.list(marker=CONNECTOR1['uuid'])
        expect = [('GET', '/v1/volume/connectors/?marker=%s' % CONNECTOR1['uuid'], {}, None)]
        expect_connectors = [CONNECTOR2]
        self._validate_list(expect, expect_connectors, volume_connectors)

    def test_volume_connectors_list_pagination_no_limit(self):
        volume_connectors = self.mgr.list(limit=0)
        expect = [('GET', '/v1/volume/connectors', {}, None), ('GET', '/v1/volume/connectors/?marker=%s' % CONNECTOR1['uuid'], {}, None)]
        expect_connectors = [CONNECTOR1, CONNECTOR2]
        self._validate_list(expect, expect_connectors, volume_connectors)