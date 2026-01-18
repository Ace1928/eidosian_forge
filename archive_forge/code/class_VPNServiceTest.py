import copy
from unittest import mock
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import vpnservice
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class VPNServiceTest(common.HeatTestCase):
    VPN_SERVICE_CONF = {'vpnservice': {'name': 'VPNService', 'description': 'My new VPN service', 'admin_state_up': True, 'router_id': 'rou123', 'subnet_id': 'sub123'}}

    def setUp(self):
        super(VPNServiceTest, self).setUp()
        self.mockclient = mock.Mock(spec=neutronclient.Client)
        self.patchobject(neutronclient, 'Client', return_value=self.mockclient)

        def lookup(client, lookup_type, name, cmd_resource):
            return name
        self.patchobject(neutronV20, 'find_resourceid_by_name_or_id', side_effect=lookup)
        self.patchobject(neutron.NeutronClientPlugin, 'has_extension', return_value=True)

    def create_vpnservice(self, resolve_neutron=True, resolve_router=True):
        self.stub_SubnetConstraint_validate()
        self.stub_RouterConstraint_validate()
        if resolve_neutron:
            snippet = template_format.parse(vpnservice_template)
        else:
            snippet = template_format.parse(vpnservice_template_deprecated)
        if resolve_router:
            props = snippet['resources']['VPNService']['properties']
            props['router'] = 'rou123'
            del props['router_id']
        self.mockclient.create_vpnservice.return_value = {'vpnservice': {'id': 'vpn123'}}
        self.stack = utils.parse_stack(snippet)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        return vpnservice.VPNService('vpnservice', resource_defns['VPNService'], self.stack)

    def test_create_deprecated(self):
        self._test_create(resolve_neutron=False)

    def test_create(self):
        self._test_create()

    def test_create_router_id(self):
        self._test_create(resolve_router=False)

    def _test_create(self, resolve_neutron=True, resolve_router=True):
        rsrc = self.create_vpnservice(resolve_neutron, resolve_router)
        self.mockclient.show_vpnservice.return_value = {'vpnservice': {'status': 'ACTIVE'}}
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        if not resolve_router:
            self.assertEqual('rou123', rsrc.properties.get(rsrc.ROUTER))
            self.assertIsNone(rsrc.properties.get(rsrc.ROUTER_ID))
        self.mockclient.create_vpnservice.assert_called_once_with(self.VPN_SERVICE_CONF)
        self.mockclient.show_vpnservice.assert_called_once_with('vpn123')

    def test_create_failed_error_status(self):
        cfg.CONF.set_override('action_retry_limit', 0)
        rsrc = self.create_vpnservice()
        self.mockclient.show_vpnservice.side_effect = [{'vpnservice': {'status': 'PENDING_CREATE'}}, {'vpnservice': {'status': 'ERROR'}}]
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        self.assertEqual('ResourceInError: resources.vpnservice: Went to status ERROR due to "Error in VPNService"', str(error))
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_vpnservice.assert_called_once_with(self.VPN_SERVICE_CONF)
        self.mockclient.show_vpnservice.assert_called_with('vpn123')

    def test_create_failed(self):
        self.stub_RouterConstraint_validate()
        self.mockclient.create_vpnservice.side_effect = exceptions.NeutronClientException
        snippet = template_format.parse(vpnservice_template)
        self.stack = utils.parse_stack(snippet)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        rsrc = vpnservice.VPNService('vpnservice', resource_defns['VPNService'], self.stack)
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        self.assertEqual('NeutronClientException: resources.vpnservice: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_vpnservice.assert_called_once_with(self.VPN_SERVICE_CONF)

    def test_delete(self):
        rsrc = self.create_vpnservice()
        self.mockclient.show_vpnservice.side_effect = [{'vpnservice': {'status': 'ACTIVE'}}, exceptions.NeutronClientException(status_code=404)]
        self.mockclient.delete_vpnservice.return_value = None
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_vpnservice.assert_called_once_with(self.VPN_SERVICE_CONF)
        self.mockclient.delete_vpnservice.assert_called_once_with('vpn123')
        self.mockclient.show_vpnservice.assert_called_with('vpn123')
        self.assertEqual(2, self.mockclient.show_vpnservice.call_count)

    def test_delete_already_gone(self):
        rsrc = self.create_vpnservice()
        self.mockclient.show_vpnservice.return_value = {'vpnservice': {'status': 'ACTIVE'}}
        self.mockclient.delete_vpnservice.side_effect = exceptions.NeutronClientException(status_code=404)
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_vpnservice.assert_called_once_with(self.VPN_SERVICE_CONF)
        self.mockclient.show_vpnservice.assert_called_once_with('vpn123')
        self.mockclient.delete_vpnservice.assert_called_once_with('vpn123')

    def test_delete_failed(self):
        rsrc = self.create_vpnservice()
        self.mockclient.show_vpnservice.return_value = {'vpnservice': {'status': 'ACTIVE'}}
        self.mockclient.delete_vpnservice.side_effect = exceptions.NeutronClientException(status_code=400)
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.delete))
        self.assertEqual('NeutronClientException: resources.vpnservice: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.DELETE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_vpnservice.assert_called_once_with(self.VPN_SERVICE_CONF)
        self.mockclient.show_vpnservice.assert_called_once_with('vpn123')
        self.mockclient.delete_vpnservice.assert_called_once_with('vpn123')

    def test_attribute(self):
        rsrc = self.create_vpnservice()
        self.mockclient.show_vpnservice.return_value = {'vpnservice': {'status': 'ACTIVE'}}
        scheduler.TaskRunner(rsrc.create)()
        self.mockclient.show_vpnservice.return_value = self.VPN_SERVICE_CONF
        self.assertEqual('VPNService', rsrc.FnGetAtt('name'))
        self.assertEqual('My new VPN service', rsrc.FnGetAtt('description'))
        self.assertIs(True, rsrc.FnGetAtt('admin_state_up'))
        self.assertEqual('rou123', rsrc.FnGetAtt('router_id'))
        self.assertEqual('sub123', rsrc.FnGetAtt('subnet_id'))
        self.mockclient.create_vpnservice.assert_called_once_with(self.VPN_SERVICE_CONF)
        self.mockclient.show_vpnservice.assert_called_with('vpn123')

    def test_attribute_failed(self):
        rsrc = self.create_vpnservice()
        self.mockclient.show_vpnservice.return_value = {'vpnservice': {'status': 'ACTIVE'}}
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'non-existent_property')
        self.assertEqual('The Referenced Attribute (vpnservice non-existent_property) is incorrect.', str(error))
        self.mockclient.create_vpnservice.assert_called_once_with(self.VPN_SERVICE_CONF)
        self.mockclient.show_vpnservice.assert_called_once_with('vpn123')

    def test_update(self):
        rsrc = self.create_vpnservice()
        self.mockclient.show_vpnservice.return_value = {'vpnservice': {'status': 'ACTIVE'}}
        self.mockclient.update_vpnservice.return_value = None
        rsrc.physical_resource_name = mock.Mock(return_value='VPNService')
        scheduler.TaskRunner(rsrc.create)()
        prop_diff = {'name': 'VPNService', 'admin_state_up': False}
        self.assertIsNone(rsrc.handle_update({}, {}, prop_diff))
        prop_diff = {'name': None, 'admin_state_up': False}
        self.assertIsNone(rsrc.handle_update({}, {}, prop_diff))
        self.mockclient.create_vpnservice.assert_called_once_with(self.VPN_SERVICE_CONF)
        self.mockclient.show_vpnservice.assert_called_once_with('vpn123')
        upd_dict = {'vpnservice': {'name': 'VPNService', 'admin_state_up': False}}
        self.mockclient.update_vpnservice.assert_called_with('vpn123', upd_dict)