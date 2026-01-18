import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class ResourceGroupTest(common.HeatTestCase):

    def setUp(self):
        super(ResourceGroupTest, self).setUp()
        self.inspector = mock.Mock(spec=grouputils.GroupInspector)
        self.patchobject(grouputils.GroupInspector, 'from_parent_resource', return_value=self.inspector)

    def test_assemble_nested(self):
        """Tests nested stack creation based on props.

        Tests that the nested stack that implements the group is created
        appropriately based on properties.
        """
        stack = utils.parse_stack(template)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        templ = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'Foo': 'Bar'}}, '1': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'Foo': 'Bar'}}, '2': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'Foo': 'Bar'}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}, '1': {'get_resource': '1'}, '2': {'get_resource': '2'}}}}}
        self.assertEqual(templ, resg._assemble_nested(['0', '1', '2']).t)

    def test_assemble_nested_outputs(self):
        """Tests nested stack creation based on props.

        Tests that the nested stack that implements the group is created
        appropriately based on properties.
        """
        stack = utils.parse_stack(template)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        templ = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'Foo': 'Bar'}}, '1': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'Foo': 'Bar'}}, '2': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'Foo': 'Bar'}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}, '1': {'get_resource': '1'}, '2': {'get_resource': '2'}}}, 'foo': {'value': [{'get_attr': ['0', 'foo']}, {'get_attr': ['1', 'foo']}, {'get_attr': ['2', 'foo']}]}}}
        resg.referenced_attrs = mock.Mock(return_value=['foo'])
        self.assertEqual(templ, resg._assemble_nested(['0', '1', '2']).t)

    def test_assemble_nested_include(self):
        templ = copy.deepcopy(template)
        res_def = templ['resources']['group1']['properties']['resource_def']
        res_def['properties']['Foo'] = None
        stack = utils.parse_stack(templ)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}}}}}
        self.assertEqual(expect, resg._assemble_nested(['0']).t)
        expect['resources']['0']['properties'] = {'Foo': None}
        self.assertEqual(expect, resg._assemble_nested(['0'], include_all=True).t)

    def test_assemble_nested_include_zero(self):
        templ = copy.deepcopy(template)
        templ['resources']['group1']['properties']['count'] = 0
        stack = utils.parse_stack(templ)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        expect = {'heat_template_version': '2015-04-30', 'outputs': {'refs_map': {'value': {}}}}
        self.assertEqual(expect, resg._assemble_nested([]).t)

    def test_assemble_nested_with_metadata(self):
        templ = copy.deepcopy(template)
        res_def = templ['resources']['group1']['properties']['resource_def']
        res_def['properties']['Foo'] = None
        res_def['metadata'] = {'priority': 'low', 'role': 'webserver'}
        stack = utils.parse_stack(templ)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {}, 'metadata': {'priority': 'low', 'role': 'webserver'}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}}}}}
        self.assertEqual(expect, resg._assemble_nested(['0']).t)

    def test_assemble_nested_rolling_update(self):
        expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'foo': 'bar'}}, '1': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'foo': 'baz'}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}, '1': {'get_resource': '1'}}}}}
        resource_def = rsrc_defn.ResourceDefinition(None, 'OverwrittenFnGetRefIdType', {'foo': 'baz'})
        stack = utils.parse_stack(template)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        nested = get_fake_nested_stack(['0', '1'])
        self.inspector.template.return_value = nested.defn._template
        self.inspector.member_names.return_value = ['0', '1']
        resg.build_resource_definition = mock.Mock(return_value=resource_def)
        self.assertEqual(expect, resg._assemble_for_rolling_update(2, 1).t)

    def test_assemble_nested_rolling_update_outputs(self):
        expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'foo': 'bar'}}, '1': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'foo': 'baz'}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}, '1': {'get_resource': '1'}}}, 'bar': {'value': [{'get_attr': ['0', 'bar']}, {'get_attr': ['1', 'bar']}]}}}
        resource_def = rsrc_defn.ResourceDefinition(None, 'OverwrittenFnGetRefIdType', {'foo': 'baz'})
        stack = utils.parse_stack(template)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        nested = get_fake_nested_stack(['0', '1'])
        self.inspector.template.return_value = nested.defn._template
        self.inspector.member_names.return_value = ['0', '1']
        resg.build_resource_definition = mock.Mock(return_value=resource_def)
        resg.referenced_attrs = mock.Mock(return_value=['bar'])
        self.assertEqual(expect, resg._assemble_for_rolling_update(2, 1).t)

    def test_assemble_nested_rolling_update_none(self):
        expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'foo': 'bar'}}, '1': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'foo': 'bar'}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}, '1': {'get_resource': '1'}}}}}
        resource_def = rsrc_defn.ResourceDefinition(None, 'OverwrittenFnGetRefIdType', {'foo': 'baz'})
        stack = utils.parse_stack(template)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        nested = get_fake_nested_stack(['0', '1'])
        self.inspector.template.return_value = nested.defn._template
        self.inspector.member_names.return_value = ['0', '1']
        resg.build_resource_definition = mock.Mock(return_value=resource_def)
        self.assertEqual(expect, resg._assemble_for_rolling_update(2, 0).t)

    def test_assemble_nested_rolling_update_failed_resource(self):
        expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'foo': 'baz'}}, '1': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'foo': 'bar'}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}, '1': {'get_resource': '1'}}}}}
        resource_def = rsrc_defn.ResourceDefinition(None, 'OverwrittenFnGetRefIdType', {'foo': 'baz'})
        stack = utils.parse_stack(template)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        nested = get_fake_nested_stack(['0', '1'])
        self.inspector.template.return_value = nested.defn._template
        self.inspector.member_names.return_value = ['1']
        resg.build_resource_definition = mock.Mock(return_value=resource_def)
        self.assertEqual(expect, resg._assemble_for_rolling_update(2, 1).t)

    def test_assemble_nested_missing_param(self):
        templ = copy.deepcopy(template)
        res_def = templ['resources']['group1']['properties']['resource_def']
        res_def['properties']['Foo'] = {'get_param': 'bar'}
        stack = utils.parse_stack(templ)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        nested_tmpl = resg._assemble_nested(['0', '1'])
        expected = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {}}, '1': {'type': 'OverwrittenFnGetRefIdType', 'properties': {}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}, '1': {'get_resource': '1'}}}}}
        self.assertEqual(expected, nested_tmpl.t)

    def test_index_var(self):
        stack = utils.parse_stack(template_repl)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'ResourceWithListProp%index%', 'properties': {'Foo': 'Bar_0', 'listprop': ['0_0', '0_1', '0_2']}}, '1': {'type': 'ResourceWithListProp%index%', 'properties': {'Foo': 'Bar_1', 'listprop': ['1_0', '1_1', '1_2']}}, '2': {'type': 'ResourceWithListProp%index%', 'properties': {'Foo': 'Bar_2', 'listprop': ['2_0', '2_1', '2_2']}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}, '1': {'get_resource': '1'}, '2': {'get_resource': '2'}}}}}
        nested = resg._assemble_nested(['0', '1', '2']).t
        for res in nested['resources']:
            res_prop = nested['resources'][res]['properties']
            res_prop['listprop'] = list(res_prop['listprop'])
        self.assertEqual(expect, nested)

    def test_custom_index_var(self):
        templ = copy.deepcopy(template_repl)
        templ['resources']['group1']['properties']['index_var'] = '__foo__'
        stack = utils.parse_stack(templ)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'ResourceWithListProp%index%', 'properties': {'Foo': 'Bar_%index%', 'listprop': ['%index%_0', '%index%_1', '%index%_2']}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}}}}}
        nested = resg._assemble_nested(['0']).t
        res_prop = nested['resources']['0']['properties']
        res_prop['listprop'] = list(res_prop['listprop'])
        self.assertEqual(expect, nested)
        props = copy.deepcopy(templ['resources']['group1']['properties'])
        res_def = props['resource_def']
        res_def['properties']['Foo'] = 'Bar___foo__'
        res_def['properties']['listprop'] = ['__foo___0', '__foo___1', '__foo___2']
        res_def['type'] = 'ResourceWithListProp__foo__'
        snip = snip.freeze(properties=props)
        resg = resource_group.ResourceGroup('test', snip, stack)
        expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'ResourceWithListProp__foo__', 'properties': {'Foo': 'Bar_0', 'listprop': ['0_0', '0_1', '0_2']}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}}}}}
        nested = resg._assemble_nested(['0']).t
        res_prop = nested['resources']['0']['properties']
        res_prop['listprop'] = list(res_prop['listprop'])
        self.assertEqual(expect, nested)

    def test_assemble_no_properties(self):
        templ = copy.deepcopy(template)
        res_def = templ['resources']['group1']['properties']['resource_def']
        del res_def['properties']
        stack = utils.parse_stack(templ)
        resg = stack.resources['group1']
        self.assertIsNone(resg.validate())

    def test_validate_with_skiplist(self):
        templ = copy.deepcopy(template_server)
        self.mock_flavor = mock.Mock(ram=4, disk=4)
        self.mock_active_image = mock.Mock(min_ram=1, min_disk=1, status='active')
        self.mock_inactive_image = mock.Mock(min_ram=1, min_disk=1, status='inactive')

        def get_image(image_identifier):
            if image_identifier == 'image0':
                return self.mock_inactive_image
            else:
                return self.mock_active_image
        self.patchobject(glance.GlanceClientPlugin, 'get_image', side_effect=get_image)
        self.patchobject(nova.NovaClientPlugin, 'get_flavor', return_value=self.mock_flavor)
        props = templ['resources']['group1']['properties']
        props['removal_policies'] = [{'resource_list': ['0']}]
        stack = utils.parse_stack(templ)
        resg = stack.resources['group1']
        self.assertIsNone(resg.validate())

    def test_invalid_res_type(self):
        """Test that error raised for unknown resource type."""
        tmp = copy.deepcopy(template)
        grp_props = tmp['resources']['group1']['properties']
        grp_props['resource_def']['type'] = 'idontexist'
        stack = utils.parse_stack(tmp)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        exc = self.assertRaises(exception.StackValidationFailed, resg.validate)
        exp_msg = 'The Resource Type (idontexist) could not be found.'
        self.assertIn(exp_msg, str(exc))

    def test_reference_attr(self):
        stack = utils.parse_stack(template2)
        snip = stack.t.resource_definitions(stack)['group1']
        resgrp = resource_group.ResourceGroup('test', snip, stack)
        self.assertIsNone(resgrp.validate())

    def test_validate_reference_attr_with_none_ref(self):
        stack = utils.parse_stack(template_attr)
        snip = stack.t.resource_definitions(stack)['group1']
        resgrp = resource_group.ResourceGroup('test', snip, stack)
        self.patchobject(resgrp, 'referenced_attrs', return_value=set([('nested_dict', None)]))
        self.assertIsNone(resgrp.validate())

    def test_invalid_removal_policies_nolist(self):
        """Test that error raised for malformed removal_policies."""
        tmp = copy.deepcopy(template)
        grp_props = tmp['resources']['group1']['properties']
        grp_props['removal_policies'] = 'notallowed'
        stack = utils.parse_stack(tmp)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        exc = self.assertRaises(exception.StackValidationFailed, resg.validate)
        errstr = 'removal_policies: "\'notallowed\'" is not a list'
        self.assertIn(errstr, str(exc))

    def test_invalid_removal_policies_nomap(self):
        """Test that error raised for malformed removal_policies."""
        tmp = copy.deepcopy(template)
        grp_props = tmp['resources']['group1']['properties']
        grp_props['removal_policies'] = ['notallowed']
        stack = utils.parse_stack(tmp)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        exc = self.assertRaises(exception.StackValidationFailed, resg.validate)
        errstr = '"notallowed" is not a map'
        self.assertIn(errstr, str(exc))

    def test_child_template(self):
        stack = utils.parse_stack(template2)
        snip = stack.t.resource_definitions(stack)['group1']

        def check_res_names(names):
            self.assertEqual(list(names), ['0', '1'])
            return 'tmpl'
        resgrp = resource_group.ResourceGroup('test', snip, stack)
        resgrp._assemble_nested = mock.Mock()
        resgrp._assemble_nested.side_effect = check_res_names
        resgrp.properties.data[resgrp.COUNT] = 2
        self.assertEqual('tmpl', resgrp.child_template())
        self.assertEqual(1, resgrp._assemble_nested.call_count)

    def test_child_params(self):
        stack = utils.parse_stack(template2)
        snip = stack.t.resource_definitions(stack)['group1']
        resgrp = resource_group.ResourceGroup('test', snip, stack)
        self.assertEqual({}, resgrp.child_params())

    def test_handle_create(self):
        stack = utils.parse_stack(template2)
        snip = stack.t.resource_definitions(stack)['group1']
        resgrp = resource_group.ResourceGroup('test', snip, stack)
        resgrp.create_with_template = mock.Mock(return_value=None)
        self.assertIsNone(resgrp.handle_create())
        self.assertEqual(1, resgrp.create_with_template.call_count)

    def test_handle_create_with_batching(self):
        self.inspector.member_names.return_value = []
        self.inspector.size.return_value = 0
        stack = utils.parse_stack(tmpl_with_default_updt_policy())
        defn = stack.t.resource_definitions(stack)['group1']
        props = stack.t.t['resources']['group1']['properties'].copy()
        props['count'] = 10
        update_policy = {'batch_create': {'max_batch_size': 3}}
        snip = defn.freeze(properties=props, update_policy=update_policy)
        resgrp = resource_group.ResourceGroup('test', snip, stack)
        self.patchobject(scheduler.TaskRunner, 'start')
        checkers = resgrp.handle_create()
        self.assertEqual(4, len(checkers))

    def test_handle_create_with_batching_zero_count(self):
        self.inspector.member_names.return_value = []
        self.inspector.size.return_value = 0
        stack = utils.parse_stack(tmpl_with_default_updt_policy())
        defn = stack.t.resource_definitions(stack)['group1']
        props = stack.t.t['resources']['group1']['properties'].copy()
        props['count'] = 0
        update_policy = {'batch_create': {'max_batch_size': 1}}
        snip = defn.freeze(properties=props, update_policy=update_policy)
        resgrp = resource_group.ResourceGroup('test', snip, stack)
        resgrp.create_with_template = mock.Mock(return_value=None)
        self.assertIsNone(resgrp.handle_create())
        self.assertEqual(1, resgrp.create_with_template.call_count)

    def test_run_to_completion(self):
        stack = utils.parse_stack(template2)
        snip = stack.t.resource_definitions(stack)['group1']
        resgrp = resource_group.ResourceGroup('test', snip, stack)
        resgrp._check_status_complete = mock.Mock(side_effect=[False, True])
        resgrp.update_with_template = mock.Mock(return_value=None)
        next(resgrp._run_to_completion(snip, 200))
        self.assertEqual(1, resgrp.update_with_template.call_count)

    def test_update_in_failed(self):
        stack = utils.parse_stack(template2)
        snip = stack.t.resource_definitions(stack)['group1']
        resgrp = resource_group.ResourceGroup('test', snip, stack)
        resgrp.state_set('CREATE', 'FAILED')
        resgrp._assemble_nested = mock.Mock(return_value='tmpl')
        resgrp.properties.data[resgrp.COUNT] = 2
        self.patchobject(scheduler.TaskRunner, 'start')
        resgrp.handle_update(snip, mock.Mock(), {})
        self.assertTrue(resgrp._assemble_nested.called)

    def test_handle_delete(self):
        stack = utils.parse_stack(template2)
        snip = stack.t.resource_definitions(stack)['group1']
        resgrp = resource_group.ResourceGroup('test', snip, stack)
        resgrp.delete_nested = mock.Mock(return_value=None)
        resgrp.handle_delete()
        resgrp.delete_nested.assert_called_once_with()

    def test_handle_update_size(self):
        stack = utils.parse_stack(template2)
        snip = stack.t.resource_definitions(stack)['group1']
        resgrp = resource_group.ResourceGroup('test', snip, stack)
        resgrp._assemble_nested = mock.Mock(return_value=None)
        resgrp.properties.data[resgrp.COUNT] = 5
        self.patchobject(scheduler.TaskRunner, 'start')
        resgrp.handle_update(snip, mock.Mock(), {})
        self.assertTrue(resgrp._assemble_nested.called)