import copy
from unittest import mock
from neutronclient.common import exceptions as q_exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.ec2 import eip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class EIPTest(common.HeatTestCase):

    def setUp(self):
        super(EIPTest, self).setUp()
        self.fc = fakes_nova.FakeClient()
        self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
        self.mock_list_net = self.patchobject(neutronclient.Client, 'list_networks')
        self.mock_create_fip = self.patchobject(neutronclient.Client, 'create_floatingip')
        self.mock_show_fip = self.patchobject(neutronclient.Client, 'show_floatingip')
        self.patchobject(neutronclient.Client, 'update_floatingip')
        self.patchobject(neutronclient.Client, 'delete_floatingip')
        self.mock_list_fips = self.patchobject(neutronclient.Client, 'list_floatingips')

    def mock_interface(self, port, ip):

        class MockIface(object):

            def __init__(self, port_id, fixed_ip):
                self.port_id = port_id
                self.fixed_ips = [{'ip_address': fixed_ip}]
        return MockIface(port, ip)

    def mock_list_floatingips(self):
        self.mock_list_fips.return_value = {'floatingips': [{'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}]}

    def mock_create_floatingip(self):
        self.mock_list_net.return_value = {'networks': [{'status': 'ACTIVE', 'subnets': [], 'name': 'nova', 'router:external': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'admin_state_up': True, 'shared': True, 'id': 'eeee'}]}
        self.mock_create_fip.return_value = {'floatingip': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'floating_ip_address': '11.0.0.1'}}

    def mock_show_floatingip(self):
        self.mock_show_fip.return_value = {'floatingip': {'router_id': None, 'tenant_id': 'e936e6cd3e0b48dcb9ff853a8f253257', 'floating_network_id': 'eeee', 'fixed_ip_address': None, 'floating_ip_address': '11.0.0.1', 'port_id': None, 'id': 'ffff'}}

    def create_eip(self, t, stack, resource_name):
        resource_defns = stack.t.resource_definitions(stack)
        rsrc = eip.ElasticIp(resource_name, resource_defns[resource_name], stack)
        self.assertIsNone(rsrc.validate())
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        stk_defn.update_resource_data(stack.defn, resource_name, rsrc.node_data())
        return rsrc

    def create_association(self, t, stack, resource_name):
        resource_defns = stack.t.resource_definitions(stack)
        rsrc = eip.ElasticIpAssociation(resource_name, resource_defns[resource_name], stack)
        self.assertIsNone(rsrc.validate())
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        stk_defn.update_resource_data(stack.defn, resource_name, rsrc.node_data())
        return rsrc

    def test_eip(self):
        mock_server = self.fc.servers.list()[0]
        self.patchobject(self.fc.servers, 'get', return_value=mock_server)
        self.mock_create_floatingip()
        iface = self.mock_interface('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '1.2.3.4')
        self.patchobject(mock_server, 'interface_list', return_value=[iface])
        t = template_format.parse(eip_template)
        stack = utils.parse_stack(t)
        rsrc = self.create_eip(t, stack, 'IPAddress')
        try:
            self.assertEqual('11.0.0.1', rsrc.FnGetRefId())
            rsrc.refid = None
            self.assertEqual('11.0.0.1', rsrc.FnGetRefId())
            self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', rsrc.FnGetAtt('AllocationId'))
            self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'Foo')
        finally:
            scheduler.TaskRunner(rsrc.destroy)()

    def test_eip_update(self):
        server_old = self.fc.servers.list()[0]
        self.patchobject(self.fc.servers, 'get', return_value=server_old)
        iface = self.mock_interface('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '1.2.3.4')
        self.patchobject(server_old, 'interface_list', return_value=[iface])
        self.mock_create_floatingip()
        t = template_format.parse(eip_template)
        stack = utils.parse_stack(t)
        rsrc = self.create_eip(t, stack, 'IPAddress')
        self.assertEqual('11.0.0.1', rsrc.FnGetRefId())
        server_update = self.fc.servers.list()[1]
        self.patchobject(self.fc.servers, 'get', return_value=server_update)
        self.patchobject(server_update, 'interface_list', return_value=[iface])
        props = copy.deepcopy(rsrc.properties.data)
        update_server_id = '5678'
        props['InstanceId'] = update_server_id
        update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
        scheduler.TaskRunner(rsrc.update, update_snippet)()
        self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
        self.assertEqual('11.0.0.1', rsrc.FnGetRefId())
        props = copy.deepcopy(rsrc.properties.data)
        props.pop('InstanceId')
        update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
        scheduler.TaskRunner(rsrc.update, update_snippet)()
        self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)

    def test_association_eip(self):
        mock_server = self.fc.servers.list()[0]
        self.patchobject(self.fc.servers, 'get', return_value=mock_server)
        iface = self.mock_interface('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '1.2.3.4')
        self.patchobject(mock_server, 'interface_list', return_value=[iface])
        self.mock_create_floatingip()
        self.mock_show_floatingip()
        self.mock_list_floatingips()
        t = template_format.parse(eip_template_ipassoc)
        stack = utils.parse_stack(t)
        rsrc = self.create_eip(t, stack, 'IPAddress')
        association = self.create_association(t, stack, 'IPAssoc')
        try:
            self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
            self.assertEqual((association.CREATE, association.COMPLETE), association.state)
            self.assertEqual(utils.PhysName(stack.name, association.name), association.FnGetRefId())
            self.assertEqual('11.0.0.1', association.properties['EIP'])
        finally:
            scheduler.TaskRunner(association.delete)()
            scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.assertEqual((association.DELETE, association.COMPLETE), association.state)

    def test_eip_with_exception(self):
        self.mock_list_net.return_value = {'networks': [{'status': 'ACTIVE', 'subnets': [], 'name': 'nova', 'router:external': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'admin_state_up': True, 'shared': True, 'id': 'eeee'}]}
        self.patchobject(neutronclient.Client, 'create_floatingip', side_effect=neutronclient.exceptions.NotFound)
        t = template_format.parse(eip_template)
        stack = utils.parse_stack(t)
        resource_name = 'IPAddress'
        resource_defns = stack.t.resource_definitions(stack)
        rsrc = eip.ElasticIp(resource_name, resource_defns[resource_name], stack)
        self.assertRaises(neutronclient.exceptions.NotFound, rsrc.handle_create)

    @mock.patch.object(eip.ElasticIp, '_ipaddress')
    def test_FnGetRefId_resource_name(self, mock_ipaddr):
        t = template_format.parse(ipassoc_template_validate)
        stack = utils.parse_stack(t)
        rsrc = stack['eip']
        mock_ipaddr.return_value = None
        self.assertEqual('eip', rsrc.FnGetRefId())

    @mock.patch.object(eip.ElasticIp, '_ipaddress')
    def test_FnGetRefId_resource_ip(self, mock_ipaddr):
        t = template_format.parse(ipassoc_template_validate)
        stack = utils.parse_stack(t)
        rsrc = stack['eip']
        mock_ipaddr.return_value = 'x.x.x.x'
        self.assertEqual('x.x.x.x', rsrc.FnGetRefId())

    def test_FnGetRefId_convergence_cache_data(self):
        t = template_format.parse(ipassoc_template_validate)
        template = tmpl.Template(t)
        stack = parser.Stack(utils.dummy_context(), 'test', template, cache_data={'eip': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': '1.1.1.1'})})
        rsrc = stack.defn['eip']
        self.assertEqual('1.1.1.1', rsrc.FnGetRefId())