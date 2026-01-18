import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.chassis
class ChassisManagerTest(testtools.TestCase):

    def setUp(self):
        super(ChassisManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)

    def test_chassis_list(self):
        chassis = self.mgr.list()
        expect = [('GET', '/v1/chassis', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(chassis))

    def test_chassis_list_limit(self):
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
        chassis = self.mgr.list(limit=1)
        expect = [('GET', '/v1/chassis/?limit=1', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(chassis, HasLength(1))

    def test_chassis_list_marker(self):
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
        chassis = self.mgr.list(marker=CHASSIS['uuid'])
        expect = [('GET', '/v1/chassis/?marker=%s' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(chassis, HasLength(1))

    def test_chassis_list_pagination_no_limit(self):
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
        chassis = self.mgr.list(limit=0)
        expect = [('GET', '/v1/chassis', {}, None), ('GET', '/v1/chassis/?limit=1', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(chassis, HasLength(2))

    def test_chassis_list_sort_key(self):
        self.api = utils.FakeAPI(fake_responses_sorting)
        self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
        chassis = self.mgr.list(sort_key='updated_at')
        expect = [('GET', '/v1/chassis/?sort_key=updated_at', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(chassis, HasLength(1))

    def test_chassis_list_sort_dir(self):
        self.api = utils.FakeAPI(fake_responses_sorting)
        self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
        chassis = self.mgr.list(sort_dir='desc')
        expect = [('GET', '/v1/chassis/?sort_dir=desc', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(chassis, HasLength(1))

    def test_chassis_list_detail(self):
        chassis = self.mgr.list(detail=True)
        expect = [('GET', '/v1/chassis/detail', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(chassis))

    def test_chassis_list_fields(self):
        nodes = self.mgr.list(fields=['uuid', 'extra'])
        expect = [('GET', '/v1/chassis/?fields=uuid,extra', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(nodes))

    def test_chassis_list_detail_and_fields_fail(self):
        self.assertRaises(exc.InvalidAttribute, self.mgr.list, detail=True, fields=['uuid', 'extra'])

    def test_chassis_show(self):
        chassis = self.mgr.get(CHASSIS['uuid'])
        expect = [('GET', '/v1/chassis/%s' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(CHASSIS['uuid'], chassis.uuid)
        self.assertEqual(CHASSIS['description'], chassis.description)

    def test_chassis_show_fields(self):
        chassis = self.mgr.get(CHASSIS['uuid'], fields=['uuid', 'description'])
        expect = [('GET', '/v1/chassis/%s?fields=uuid,description' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(CHASSIS['uuid'], chassis.uuid)
        self.assertEqual(CHASSIS['description'], chassis.description)

    def test_create(self):
        chassis = self.mgr.create(**CREATE_CHASSIS)
        expect = [('POST', '/v1/chassis', {}, CREATE_CHASSIS)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(chassis)

    def test_create_with_uuid(self):
        chassis = self.mgr.create(**CREATE_WITH_UUID)
        expect = [('POST', '/v1/chassis', {}, CREATE_WITH_UUID)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(chassis)

    def test_delete(self):
        chassis = self.mgr.delete(chassis_id=CHASSIS['uuid'])
        expect = [('DELETE', '/v1/chassis/%s' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(chassis)

    def test_update(self):
        patch = {'op': 'replace', 'value': NEW_DESCR, 'path': '/description'}
        chassis = self.mgr.update(chassis_id=CHASSIS['uuid'], patch=patch)
        expect = [('PATCH', '/v1/chassis/%s' % CHASSIS['uuid'], {}, patch)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(NEW_DESCR, chassis.description)

    def test_chassis_node_list(self):
        nodes = self.mgr.list_nodes(CHASSIS['uuid'])
        expect = [('GET', '/v1/chassis/%s/nodes' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(nodes))
        self.assertEqual(NODE['uuid'], nodes[0].uuid)

    def test_chassis_node_list_detail(self):
        nodes = self.mgr.list_nodes(CHASSIS['uuid'], detail=True)
        expect = [('GET', '/v1/chassis/%s/nodes/detail' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(nodes))
        self.assertEqual(NODE['uuid'], nodes[0].uuid)

    def test_chassis_node_list_fields(self):
        nodes = self.mgr.list_nodes(CHASSIS['uuid'], fields=['uuid', 'extra'])
        expect = [('GET', '/v1/chassis/%s/nodes?fields=uuid,extra' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(nodes))

    def test_chassis_node_list_maintenance(self):
        nodes = self.mgr.list_nodes(CHASSIS['uuid'], maintenance=False)
        expect = [('GET', '/v1/chassis/%s/nodes?maintenance=False' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(nodes))

    def test_chassis_node_list_associated(self):
        nodes = self.mgr.list_nodes(CHASSIS['uuid'], associated=True)
        expect = [('GET', '/v1/chassis/%s/nodes?associated=True' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(nodes))

    def test_chassis_node_list_provision_state(self):
        nodes = self.mgr.list_nodes(CHASSIS['uuid'], provision_state='available')
        expect = [('GET', '/v1/chassis/%s/nodes?provision_state=available' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(nodes))

    def test_chassis_node_list_detail_and_fields_fail(self):
        self.assertRaises(exc.InvalidAttribute, self.mgr.list_nodes, CHASSIS['uuid'], detail=True, fields=['uuid', 'extra'])

    def test_chassis_node_list_limit(self):
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
        nodes = self.mgr.list_nodes(CHASSIS['uuid'], limit=1)
        expect = [('GET', '/v1/chassis/%s/nodes?limit=1' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(nodes, HasLength(1))
        self.assertEqual(NODE['uuid'], nodes[0].uuid)

    def test_chassis_node_list_sort_key(self):
        self.api = utils.FakeAPI(fake_responses_sorting)
        self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
        nodes = self.mgr.list_nodes(CHASSIS['uuid'], sort_key='updated_at')
        expect = [('GET', '/v1/chassis/%s/nodes?sort_key=updated_at' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(nodes, HasLength(1))
        self.assertEqual(NODE['uuid'], nodes[0].uuid)

    def test_chassis_node_list_sort_dir(self):
        self.api = utils.FakeAPI(fake_responses_sorting)
        self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
        nodes = self.mgr.list_nodes(CHASSIS['uuid'], sort_dir='desc')
        expect = [('GET', '/v1/chassis/%s/nodes?sort_dir=desc' % CHASSIS['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(nodes, HasLength(1))
        self.assertEqual(NODE['uuid'], nodes[0].uuid)

    def test_chassis_node_list_marker(self):
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
        nodes = self.mgr.list_nodes(CHASSIS['uuid'], marker=NODE['uuid'])
        expect = [('GET', '/v1/chassis/%s/nodes?marker=%s' % (CHASSIS['uuid'], NODE['uuid']), {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(nodes, HasLength(1))
        self.assertEqual(NODE['uuid'], nodes[0].uuid)