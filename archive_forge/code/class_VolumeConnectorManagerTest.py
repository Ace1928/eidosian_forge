import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
class VolumeConnectorManagerTest(VolumeConnectorManagerTestBase):

    def setUp(self):
        super(VolumeConnectorManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = ironicclient.v1.volume_connector.VolumeConnectorManager(self.api)

    def test_volume_connectors_list(self):
        volume_connectors = self.mgr.list()
        expect = [('GET', '/v1/volume/connectors', {}, None)]
        expect_connectors = [CONNECTOR1]
        self._validate_list(expect, expect_connectors, volume_connectors)

    def test_volume_connectors_list_by_node(self):
        volume_connectors = self.mgr.list(node=NODE_UUID)
        expect = [('GET', '/v1/volume/connectors/?node=%s' % NODE_UUID, {}, None)]
        expect_connectors = [CONNECTOR1]
        self._validate_list(expect, expect_connectors, volume_connectors)

    def test_volume_connectors_list_by_node_detail(self):
        volume_connectors = self.mgr.list(node=NODE_UUID, detail=True)
        expect = [('GET', '/v1/volume/connectors/?detail=True&node=%s' % NODE_UUID, {}, None)]
        expect_connectors = [CONNECTOR1]
        self._validate_list(expect, expect_connectors, volume_connectors)

    def test_volume_connectors_list_detail(self):
        volume_connectors = self.mgr.list(detail=True)
        expect = [('GET', '/v1/volume/connectors/?detail=True', {}, None)]
        expect_connectors = [CONNECTOR1]
        self._validate_list(expect, expect_connectors, volume_connectors)

    def test_volume_connector_list_fields(self):
        volume_connectors = self.mgr.list(fields=['uuid', 'connector_id'])
        expect = [('GET', '/v1/volume/connectors/?fields=uuid,connector_id', {}, None)]
        expect_connectors = [CONNECTOR1]
        self._validate_list(expect, expect_connectors, volume_connectors)

    def test_volume_connector_list_detail_and_fields_fail(self):
        self.assertRaises(exc.InvalidAttribute, self.mgr.list, detail=True, fields=['uuid', 'connector_id'])

    def test_volume_connectors_show(self):
        volume_connector = self.mgr.get(CONNECTOR1['uuid'])
        expect = [('GET', '/v1/volume/connectors/%s' % CONNECTOR1['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self._validate_obj(CONNECTOR1, volume_connector)

    def test_volume_connector_show_fields(self):
        volume_connector = self.mgr.get(CONNECTOR1['uuid'], fields=['uuid', 'connector_id'])
        expect = [('GET', '/v1/volume/connectors/%s?fields=uuid,connector_id' % CONNECTOR1['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(CONNECTOR1['uuid'], volume_connector.uuid)
        self.assertEqual(CONNECTOR1['connector_id'], volume_connector.connector_id)

    def test_create(self):
        volume_connector = self.mgr.create(**CREATE_CONNECTOR)
        expect = [('POST', '/v1/volume/connectors', {}, CREATE_CONNECTOR)]
        self.assertEqual(expect, self.api.calls)
        self._validate_obj(CONNECTOR1, volume_connector)

    def test_create_with_uuid(self):
        volume_connector = self.mgr.create(**CREATE_CONNECTOR_WITH_UUID)
        expect = [('POST', '/v1/volume/connectors', {}, CREATE_CONNECTOR_WITH_UUID)]
        self.assertEqual(expect, self.api.calls)
        self._validate_obj(CREATE_CONNECTOR_WITH_UUID, volume_connector)

    def test_delete(self):
        volume_connector = self.mgr.delete(CONNECTOR1['uuid'])
        expect = [('DELETE', '/v1/volume/connectors/%s' % CONNECTOR1['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(volume_connector)

    def test_update(self):
        patch = {'op': 'replace', 'connector_id': NEW_CONNECTOR_ID, 'path': '/connector_id'}
        volume_connector = self.mgr.update(volume_connector_id=CONNECTOR1['uuid'], patch=patch)
        expect = [('PATCH', '/v1/volume/connectors/%s' % CONNECTOR1['uuid'], {}, patch)]
        self.assertEqual(expect, self.api.calls)
        self._validate_obj(UPDATED_CONNECTOR, volume_connector)