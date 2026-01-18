from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
class TestAutoScalingPolicy(common.HeatTestCase):

    def create_scaling_policy(self, t, stack, resource_name):
        rsrc = stack[resource_name]
        self.assertIsNone(rsrc.validate())
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        return rsrc

    def test_validate_scaling_policy_ok(self):
        t = template_format.parse(as_template)
        t['resources']['my-policy']['properties']['scaling_adjustment'] = 33
        t['resources']['my-policy']['properties']['adjustment_type'] = 'percent_change_in_capacity'
        t['resources']['my-policy']['properties']['min_adjustment_step'] = 2
        stack = utils.parse_stack(t)
        self.assertIsNone(stack.validate())

    def test_validate_scaling_policy_error(self):
        t = template_format.parse(as_template)
        t['resources']['my-policy']['properties']['scaling_adjustment'] = 1
        t['resources']['my-policy']['properties']['adjustment_type'] = 'change_in_capacity'
        t['resources']['my-policy']['properties']['min_adjustment_step'] = 2
        stack = utils.parse_stack(t)
        ex = self.assertRaises(exception.ResourcePropertyValueDependency, stack.validate)
        self.assertIn('min_adjustment_step property should only be specified for adjustment_type with value percent_change_in_capacity.', str(ex))

    def test_scaling_policy_bad_group(self):
        t = template_format.parse(inline_templates.as_heat_template_bad_group)
        stack = utils.parse_stack(t)
        up_policy = self.create_scaling_policy(t, stack, 'my-policy')
        ex = self.assertRaises(exception.ResourceFailure, up_policy.signal)
        self.assertIn('Alarm my-policy could not find scaling group', str(ex))

    def test_scaling_policy_adjust_no_action(self):
        t = template_format.parse(as_template)
        stack = utils.parse_stack(t, params=as_params)
        up_policy = self.create_scaling_policy(t, stack, 'my-policy')
        group = stack['my-group']
        self.patchobject(group, 'adjust', side_effect=resource.NoActionRequired())
        self.assertRaises(resource.NoActionRequired, up_policy.handle_signal)

    def test_scaling_policy_adjust_size_changed(self):
        t = template_format.parse(as_template)
        stack = utils.parse_stack(t, params=as_params)
        up_policy = self.create_scaling_policy(t, stack, 'my-policy')
        group = stack['my-group']
        self.patchobject(group, 'resize')
        self.patchobject(group, '_lb_reload')
        mock_fin_scaling = self.patchobject(group, '_finished_scaling')
        with mock.patch.object(group, '_check_scaling_allowed') as mock_isa:
            self.assertIsNone(up_policy.handle_signal())
            mock_isa.assert_called_once_with(60)
            mock_fin_scaling.assert_called_once_with(60, 'change_in_capacity : 1', size_changed=True)

    def test_scaling_policy_cooldown_toosoon(self):
        t = template_format.parse(as_template)
        stack = utils.parse_stack(t, params=as_params)
        pol = self.create_scaling_policy(t, stack, 'my-policy')
        group = stack['my-group']
        test = {'current': 'alarm'}
        with mock.patch.object(group, '_check_scaling_allowed', side_effect=resource.NoActionRequired) as mock_cip:
            self.assertRaises(resource.NoActionRequired, pol.handle_signal, details=test)
            mock_cip.assert_called_once_with(60)

    def test_scaling_policy_cooldown_ok(self):
        t = template_format.parse(as_template)
        stack = utils.parse_stack(t, params=as_params)
        pol = self.create_scaling_policy(t, stack, 'my-policy')
        group = stack['my-group']
        test = {'current': 'alarm'}
        self.patchobject(group, '_finished_scaling')
        self.patchobject(group, '_lb_reload')
        mock_resize = self.patchobject(group, 'resize')
        with mock.patch.object(group, '_check_scaling_allowed') as mock_isa:
            pol.handle_signal(details=test)
            mock_isa.assert_called_once_with(60)
        mock_resize.assert_called_once_with(1)

    def test_scaling_policy_refid(self):
        t = template_format.parse(as_template)
        stack = utils.parse_stack(t)
        rsrc = stack['my-policy']
        rsrc.resource_id = 'xyz'
        self.assertEqual('xyz', rsrc.FnGetRefId())

    def test_scaling_policy_refid_convg_cache_data(self):
        t = template_format.parse(as_template)
        cache_data = {'my-policy': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'convg_xyz'})}
        stack = utils.parse_stack(t, cache_data=cache_data)
        rsrc = stack.defn['my-policy']
        self.assertEqual('convg_xyz', rsrc.FnGetRefId())