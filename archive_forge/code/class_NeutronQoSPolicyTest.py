from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class NeutronQoSPolicyTest(common.HeatTestCase):

    def setUp(self):
        super(NeutronQoSPolicyTest, self).setUp()
        self.ctx = utils.dummy_context()
        tpl = template_format.parse(qos_policy_template)
        self.stack = stack.Stack(self.ctx, 'neutron_qos_policy_test', template.Template(tpl))
        self.neutronclient = mock.MagicMock()
        self.patchobject(neutron.NeutronClientPlugin, 'has_extension', return_value=True)
        self.my_qos_policy = self.stack['my_qos_policy']
        self.my_qos_policy.client = mock.MagicMock(return_value=self.neutronclient)
        self.patchobject(self.my_qos_policy, 'physical_resource_name', return_value='test_policy')

    def test_qos_policy_handle_create(self):
        policy = {'policy': {'description': 'a policy for test', 'id': '9c1eb3fe-7bba-479d-bd43-1d497e53c384', 'rules': [], 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0', 'shared': True}}
        create_props = {'name': 'test_policy', 'description': 'a policy for test', 'shared': True, 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0'}
        self.neutronclient.create_qos_policy.return_value = policy
        self.my_qos_policy.handle_create()
        self.assertEqual('9c1eb3fe-7bba-479d-bd43-1d497e53c384', self.my_qos_policy.resource_id)
        self.neutronclient.create_qos_policy.assert_called_once_with({'policy': create_props})

    def test_qos_policy_handle_delete(self):
        policy_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
        self.my_qos_policy.resource_id = policy_id
        self.neutronclient.delete_qos_policy.return_value = None
        self.assertIsNone(self.my_qos_policy.handle_delete())
        self.neutronclient.delete_qos_policy.assert_called_once_with(self.my_qos_policy.resource_id)

    def test_qos_policy_handle_delete_not_found(self):
        policy_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
        self.my_qos_policy.resource_id = policy_id
        not_found = self.neutronclient.NotFound
        self.neutronclient.delete_qos_policy.side_effect = not_found
        self.assertIsNone(self.my_qos_policy.handle_delete())
        self.neutronclient.delete_qos_policy.assert_called_once_with(self.my_qos_policy.resource_id)

    def test_qos_policy_handle_delete_resource_id_is_none(self):
        self.my_qos_policy.resource_id = None
        self.assertIsNone(self.my_qos_policy.handle_delete())
        self.assertEqual(0, self.neutronclient.delete_qos_policy.call_count)

    def test_qos_policy_handle_update(self):
        policy_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
        self.my_qos_policy.resource_id = policy_id
        props = {'name': 'test_policy', 'description': 'test', 'shared': False}
        prop_dict = props.copy()
        update_snippet = rsrc_defn.ResourceDefinition(self.my_qos_policy.name, self.my_qos_policy.type(), props)
        self.my_qos_policy.handle_update(json_snippet=update_snippet, tmpl_diff={}, prop_diff=props)
        props['name'] = None
        self.my_qos_policy.handle_update(json_snippet=update_snippet, tmpl_diff={}, prop_diff=props)
        self.assertEqual(2, self.neutronclient.update_qos_policy.call_count)
        self.neutronclient.update_qos_policy.assert_called_with(policy_id, {'policy': prop_dict})

    def test_qos_policy_get_attr(self):
        self.my_qos_policy.resource_id = 'test policy'
        policy = {'policy': {'name': 'test_policy', 'description': 'a policy for test', 'id': '9c1eb3fe-7bba-479d-bd43-1d497e53c384', 'rules': [], 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0', 'shared': True}}
        self.neutronclient.show_qos_policy.return_value = policy
        self.assertEqual([], self.my_qos_policy.FnGetAtt('rules'))
        self.assertEqual(policy['policy'], self.my_qos_policy.FnGetAtt('show'))
        self.neutronclient.show_qos_policy.assert_has_calls([mock.call(self.my_qos_policy.resource_id)] * 2)