import copy
import testtools
from ironicclient.tests.unit import utils
import ironicclient.v1.portgroup
class PortgroupManagerTest(testtools.TestCase):

    def setUp(self):
        super(PortgroupManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = ironicclient.v1.portgroup.PortgroupManager(self.api)

    def test_portgroups_list(self):
        portgroups = self.mgr.list()
        expect = [('GET', '/v1/portgroups', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(2, len(portgroups))
        expected_resp = ({}, {'portgroups': [PORTGROUP, PORTGROUP2]})
        self.assertEqual(expected_resp, self.api.responses['/v1/portgroups']['GET'])

    def test_portgroups_list_by_address(self):
        portgroups = self.mgr.list(address=PORTGROUP['address'])
        expect = [('GET', '/v1/portgroups/?address=%s' % PORTGROUP['address'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(portgroups))
        expected_resp = ({}, {'portgroups': [PORTGROUP, PORTGROUP2]})
        self.assertEqual(expected_resp, self.api.responses['/v1/portgroups']['GET'])

    def test_portgroups_list_by_address_detail(self):
        portgroups = self.mgr.list(address=PORTGROUP['address'], detail=True)
        expect = [('GET', '/v1/portgroups/detail?address=%s' % PORTGROUP['address'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(portgroups))
        self.assertIn(PORTGROUP, self.api.responses['/v1/portgroups']['GET'][1]['portgroups'])

    def test_portgroups_list_detail(self):
        portgroups = self.mgr.list(detail=True)
        expect = [('GET', '/v1/portgroups/detail', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(2, len(portgroups))
        expected_resp = ({}, {'portgroups': [PORTGROUP, PORTGROUP2]})
        self.assertEqual(expected_resp, self.api.responses['/v1/portgroups']['GET'])

    def test_portgroups_show(self):
        portgroup = self.mgr.get(PORTGROUP['uuid'])
        expect = [('GET', '/v1/portgroups/%s' % PORTGROUP['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(PORTGROUP['uuid'], portgroup.uuid)
        self.assertEqual(PORTGROUP['name'], portgroup.name)
        self.assertEqual(PORTGROUP['node_uuid'], portgroup.node_uuid)
        self.assertEqual(PORTGROUP['address'], portgroup.address)
        expected_resp = ({}, PORTGROUP)
        self.assertEqual(expected_resp, self.api.responses['/v1/portgroups/%s' % PORTGROUP['uuid']]['GET'])

    def test_portgroups_show_by_address(self):
        portgroup = self.mgr.get_by_address(PORTGROUP['address'])
        expect = [('GET', '/v1/portgroups/detail?address=%s' % PORTGROUP['address'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(PORTGROUP['uuid'], portgroup.uuid)
        self.assertEqual(PORTGROUP['name'], portgroup.name)
        self.assertEqual(PORTGROUP['node_uuid'], portgroup.node_uuid)
        self.assertEqual(PORTGROUP['address'], portgroup.address)
        expected_resp = ({}, {'portgroups': [PORTGROUP]})
        self.assertEqual(expected_resp, self.api.responses['/v1/portgroups/detail?address=%s' % PORTGROUP['address']]['GET'])

    def test_create(self):
        portgroup = self.mgr.create(**CREATE_PORTGROUP)
        expect = [('POST', '/v1/portgroups', {}, CREATE_PORTGROUP)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(portgroup)
        self.assertIn(PORTGROUP, self.api.responses['/v1/portgroups']['GET'][1]['portgroups'])

    def test_delete(self):
        portgroup = self.mgr.delete(portgroup_id=PORTGROUP['uuid'])
        expect = [('DELETE', '/v1/portgroups/%s' % PORTGROUP['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(portgroup)
        expected_resp = ({}, PORTGROUP)
        self.assertEqual(expected_resp, self.api.responses['/v1/portgroups/%s' % PORTGROUP['uuid']]['GET'])

    def test_update(self):
        patch = {'op': 'replace', 'value': NEW_ADDR, 'path': '/address'}
        portgroup = self.mgr.update(portgroup_id=PORTGROUP['uuid'], patch=patch)
        expect = [('PATCH', '/v1/portgroups/%s' % PORTGROUP['uuid'], {}, patch)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(NEW_ADDR, portgroup.address)
        expected_resp = ({}, PORTGROUP)
        self.assertEqual(expected_resp, self.api.responses['/v1/portgroups/%s' % PORTGROUP['uuid']]['GET'])

    def test_portgroup_port_list_with_uuid(self):
        ports = self.mgr.list_ports(PORTGROUP['uuid'])
        expect = [('GET', '/v1/portgroups/%s/ports' % PORTGROUP['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(ports))
        self.assertEqual(PORT['uuid'], ports[0].uuid)
        self.assertEqual(PORT['address'], ports[0].address)
        expected_resp = ({}, {'ports': [PORT]})
        self.assertEqual(expected_resp, self.api.responses['/v1/portgroups/%s/ports' % PORTGROUP['uuid']]['GET'])

    def test_portgroup_port_list_with_name(self):
        ports = self.mgr.list_ports(PORTGROUP['name'])
        expect = [('GET', '/v1/portgroups/%s/ports' % PORTGROUP['name'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(ports))
        self.assertEqual(PORT['uuid'], ports[0].uuid)
        self.assertEqual(PORT['address'], ports[0].address)
        expected_resp = ({}, {'ports': [PORT]})
        self.assertEqual(expected_resp, self.api.responses['/v1/portgroups/%s/ports' % PORTGROUP['name']]['GET'])