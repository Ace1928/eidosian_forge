from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class NeutronQoSDscpMarkingRuleTest(common.HeatTestCase):

    def setUp(self):
        super(NeutronQoSDscpMarkingRuleTest, self).setUp()
        self.ctx = utils.dummy_context()
        tpl = template_format.parse(dscp_marking_rule_template)
        self.stack = stack.Stack(self.ctx, 'neutron_dscp_marking_rule_test', template.Template(tpl))
        self.neutronclient = mock.MagicMock()
        self.patchobject(neutron.NeutronClientPlugin, 'has_extension', return_value=True)
        self.dscp_marking_rule = self.stack['my_dscp_marking_rule']
        self.dscp_marking_rule.client = mock.MagicMock(return_value=self.neutronclient)
        self.find_mock = self.patchobject(neutron.neutronV20, 'find_resourceid_by_name_or_id')
        self.policy_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
        self.find_mock.return_value = self.policy_id

    def test_rule_handle_create(self):
        rule = {'dscp_marking_rule': {'id': 'cf0eab12-ef8b-4a62-98d0-70576583c17a', 'dscp_mark': 16, 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0'}}
        create_props = {'dscp_mark': 16, 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0'}
        self.neutronclient.create_dscp_marking_rule.return_value = rule
        self.dscp_marking_rule.handle_create()
        self.assertEqual('cf0eab12-ef8b-4a62-98d0-70576583c17a', self.dscp_marking_rule.resource_id)
        self.neutronclient.create_dscp_marking_rule.assert_called_once_with(self.policy_id, {'dscp_marking_rule': create_props})

    def test_rule_handle_delete(self):
        rule_id = 'cf0eab12-ef8b-4a62-98d0-70576583c17a'
        self.dscp_marking_rule.resource_id = rule_id
        self.neutronclient.delete_dscp_marking_rule.return_value = None
        self.assertIsNone(self.dscp_marking_rule.handle_delete())
        self.neutronclient.delete_dscp_marking_rule.assert_called_once_with(rule_id, self.policy_id)

    def test_rule_handle_delete_not_found(self):
        rule_id = 'cf0eab12-ef8b-4a62-98d0-70576583c17a'
        self.dscp_marking_rule.resource_id = rule_id
        not_found = self.neutronclient.NotFound
        self.neutronclient.delete_dscp_marking_rule.side_effect = not_found
        self.assertIsNone(self.dscp_marking_rule.handle_delete())
        self.neutronclient.delete_dscp_marking_rule.assert_called_once_with(rule_id, self.policy_id)

    def test_rule_handle_delete_resource_id_is_none(self):
        self.dscp_marking_rule.resource_id = None
        self.assertIsNone(self.dscp_marking_rule.handle_delete())
        self.assertEqual(0, self.neutronclient.dscp_marking_rule.call_count)

    def test_rule_handle_update(self):
        rule_id = 'cf0eab12-ef8b-4a62-98d0-70576583c17a'
        self.dscp_marking_rule.resource_id = rule_id
        prop_diff = {'dscp_mark': 8}
        self.dscp_marking_rule.handle_update(json_snippet={}, tmpl_diff={}, prop_diff=prop_diff)
        self.neutronclient.update_dscp_marking_rule.assert_called_once_with(rule_id, self.policy_id, {'dscp_marking_rule': prop_diff})

    def test_rule_get_attr(self):
        self.dscp_marking_rule.resource_id = 'test rule'
        rule = {'dscp_marking_rule': {'id': 'cf0eab12-ef8b-4a62-98d0-70576583c17a', 'dscp_mark': 8, 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0'}}
        self.neutronclient.show_dscp_marking_rule.return_value = rule
        self.assertEqual(rule['dscp_marking_rule'], self.dscp_marking_rule.FnGetAtt('show'))
        self.neutronclient.show_dscp_marking_rule.assert_called_once_with(self.dscp_marking_rule.resource_id, self.policy_id)