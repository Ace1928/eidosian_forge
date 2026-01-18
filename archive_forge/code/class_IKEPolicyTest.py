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
class IKEPolicyTest(common.HeatTestCase):
    IKE_POLICY_CONF = {'ikepolicy': {'name': 'IKEPolicy', 'description': 'My new IKE policy', 'auth_algorithm': 'sha1', 'encryption_algorithm': '3des', 'phase1_negotiation_mode': 'main', 'lifetime': {'units': 'seconds', 'value': 3600}, 'pfs': 'group5', 'ike_version': 'v1'}}

    def setUp(self):
        super(IKEPolicyTest, self).setUp()
        self.mockclient = mock.Mock(spec=neutronclient.Client)
        self.patchobject(neutronclient, 'Client', return_value=self.mockclient)
        self.patchobject(neutron.NeutronClientPlugin, 'has_extension', return_value=True)

    def create_ikepolicy(self):
        self.mockclient.create_ikepolicy.return_value = {'ikepolicy': {'id': 'ike123'}}
        snippet = template_format.parse(ikepolicy_template)
        self.stack = utils.parse_stack(snippet)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        return vpnservice.IKEPolicy('ikepolicy', resource_defns['IKEPolicy'], self.stack)

    def test_create(self):
        rsrc = self.create_ikepolicy()
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_ikepolicy.assert_called_once_with(self.IKE_POLICY_CONF)

    def test_create_failed(self):
        self.mockclient.create_ikepolicy.side_effect = exceptions.NeutronClientException
        snippet = template_format.parse(ikepolicy_template)
        self.stack = utils.parse_stack(snippet)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        rsrc = vpnservice.IKEPolicy('ikepolicy', resource_defns['IKEPolicy'], self.stack)
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        self.assertEqual('NeutronClientException: resources.ikepolicy: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_ikepolicy.assert_called_once_with(self.IKE_POLICY_CONF)

    def test_delete(self):
        rsrc = self.create_ikepolicy()
        self.mockclient.delete_ikepolicy.return_value = None
        self.mockclient.show_ikepolicy.side_effect = exceptions.NeutronClientException(status_code=404)
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_ikepolicy.assert_called_once_with(self.IKE_POLICY_CONF)
        self.mockclient.delete_ikepolicy.assert_called_once_with('ike123')
        self.mockclient.show_ikepolicy.assert_called_once_with('ike123')

    def test_delete_already_gone(self):
        rsrc = self.create_ikepolicy()
        self.mockclient.delete_ikepolicy.side_effect = exceptions.NeutronClientException(status_code=404)
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_ikepolicy.assert_called_once_with(self.IKE_POLICY_CONF)
        self.mockclient.delete_ikepolicy.assert_called_once_with('ike123')
        self.mockclient.show_ikepolicy.assert_not_called()

    def test_delete_failed(self):
        rsrc = self.create_ikepolicy()
        self.mockclient.delete_ikepolicy.side_effect = exceptions.NeutronClientException(status_code=400)
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.delete))
        self.assertEqual('NeutronClientException: resources.ikepolicy: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.DELETE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_ikepolicy.assert_called_once_with(self.IKE_POLICY_CONF)
        self.mockclient.delete_ikepolicy.assert_called_once_with('ike123')
        self.mockclient.show_ikepolicy.assert_not_called()

    def test_attribute(self):
        rsrc = self.create_ikepolicy()
        self.mockclient.show_ikepolicy.return_value = self.IKE_POLICY_CONF
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual('IKEPolicy', rsrc.FnGetAtt('name'))
        self.assertEqual('My new IKE policy', rsrc.FnGetAtt('description'))
        self.assertEqual('sha1', rsrc.FnGetAtt('auth_algorithm'))
        self.assertEqual('3des', rsrc.FnGetAtt('encryption_algorithm'))
        self.assertEqual('main', rsrc.FnGetAtt('phase1_negotiation_mode'))
        self.assertEqual('seconds', rsrc.FnGetAtt('lifetime')['units'])
        self.assertEqual(3600, rsrc.FnGetAtt('lifetime')['value'])
        self.assertEqual('group5', rsrc.FnGetAtt('pfs'))
        self.assertEqual('v1', rsrc.FnGetAtt('ike_version'))
        self.mockclient.create_ikepolicy.assert_called_once_with(self.IKE_POLICY_CONF)
        self.mockclient.show_ikepolicy.assert_called_with('ike123')

    def test_attribute_failed(self):
        rsrc = self.create_ikepolicy()
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'non-existent_property')
        self.assertEqual('The Referenced Attribute (ikepolicy non-existent_property) is incorrect.', str(error))
        self.mockclient.create_ikepolicy.assert_called_once_with(self.IKE_POLICY_CONF)
        self.mockclient.show_ikepolicy.assert_not_called()

    def test_update(self):
        rsrc = self.create_ikepolicy()
        self.mockclient.update_ikepolicy.return_value = None
        new_props = {'name': 'New IKEPolicy', 'auth_algorithm': 'sha512', 'description': 'New description', 'encryption_algorithm': 'aes-256', 'lifetime': {'units': 'seconds', 'value': 1800}, 'pfs': 'group2', 'ike_version': 'v2'}
        update_body = {'ikepolicy': new_props}
        scheduler.TaskRunner(rsrc.create)()
        props = dict(rsrc.properties)
        props.update(new_props)
        update_template = rsrc.t.freeze(properties=props)
        scheduler.TaskRunner(rsrc.update, update_template)()
        self.mockclient.create_ikepolicy.assert_called_once_with(self.IKE_POLICY_CONF)
        self.mockclient.update_ikepolicy.assert_called_once_with('ike123', update_body)