import contextlib
import json
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import exceptions as msg_exceptions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources import stack_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class StackResourceTest(StackResourceBaseTest):

    def setUp(self):
        super(StackResourceTest, self).setUp()
        self.templ = template_format.parse(param_template)
        self.simple_template = template_format.parse(simple_template)
        orig_dumps = jsonutils.dumps

        def sorted_dumps(*args, **kwargs):
            kwargs.setdefault('sort_keys', True)
            return orig_dumps(*args, **kwargs)
        patched_dumps = mock.patch('oslo_serialization.jsonutils.dumps', sorted_dumps)
        patched_dumps.start()
        self.addCleanup(lambda: patched_dumps.stop())

    def test_child_template_defaults_to_not_implemented(self):
        self.assertRaises(NotImplementedError, self.parent_resource.child_template)

    def test_child_params_defaults_to_not_implemented(self):
        self.assertRaises(NotImplementedError, self.parent_resource.child_params)

    def test_preview_defaults_to_stack_resource_itself(self):
        preview = self.parent_resource.preview()
        self.assertIsInstance(preview, stack_resource.StackResource)

    def test_nested_stack_abandon(self):
        nest = mock.MagicMock()
        self.parent_resource.nested = nest
        nest.return_value.prepare_abandon.return_value = {'X': 'Y'}
        ret = self.parent_resource.prepare_abandon()
        nest.return_value.prepare_abandon.assert_called_once_with()
        self.assertEqual({'X': 'Y'}, ret)

    def test_nested_abandon_stack_not_found(self):
        self.parent_resource.nested = mock.MagicMock(return_value=None)
        ret = self.parent_resource.prepare_abandon()
        self.assertEqual({}, ret)

    def test_abandon_nested_sends_rpc_abandon(self):
        rpcc = mock.MagicMock()

        @contextlib.contextmanager
        def exc_filter(*args):
            try:
                yield
            except exception.NotFound:
                pass
        rpcc.ignore_error_by_name.side_effect = exc_filter
        self.parent_resource.rpc_client = rpcc
        self.parent_resource.resource_id = 'fake_id'
        self.parent_resource.prepare_abandon()
        status = ('CREATE', 'COMPLETE', '', 'now_time')
        with mock.patch.object(stack_object.Stack, 'get_status', return_value=status):
            self.parent_resource.delete_nested()
        rpcc.return_value.abandon_stack.assert_called_once_with(self.parent_resource.context, mock.ANY)
        rpcc.return_value.delete_stack.assert_not_called()

    def test_propagated_files(self):
        """Test passing of the files map in the top level to the child.

        Makes sure that the files map in the top level stack are passed on to
        the child stack.
        """
        self.parent_stack.t.files['foo'] = 'bar'
        parsed_t = self.parent_resource._parse_child_template(self.templ, None)
        self.assertEqual({'foo': 'bar'}, parsed_t.files.files)

    @mock.patch('heat.engine.environment.get_child_environment')
    @mock.patch.object(stack_resource.parser, 'Stack')
    def test_preview_with_implemented_child_resource(self, mock_stack_class, mock_env_class):
        nested_stack = mock.Mock()
        mock_stack_class.return_value = nested_stack
        nested_stack.preview_resources.return_value = 'preview_nested_stack'
        mock_env_class.return_value = 'environment'
        template = templatem.Template(template_format.parse(param_template))
        parent_t = self.parent_stack.t
        resource_defns = parent_t.resource_definitions(self.parent_stack)
        parent_resource = MyImplementedStackResource('test', resource_defns[self.ws_resname], self.parent_stack)
        params = {'KeyName': 'test'}
        parent_resource.set_template(template, params)
        validation_mock = mock.Mock(return_value=None)
        parent_resource._validate_nested_resources = validation_mock
        result = parent_resource.preview()
        mock_env_class.assert_called_once_with(self.parent_stack.env, params, child_resource_name='test', item_to_remove=None)
        self.assertEqual('preview_nested_stack', result)
        mock_stack_class.assert_called_once_with(mock.ANY, 'test_stack-test', mock.ANY, timeout_mins=None, disable_rollback=True, parent_resource=parent_resource.name, owner_id=self.parent_stack.id, user_creds_id=self.parent_stack.user_creds_id, stack_user_project_id=self.parent_stack.stack_user_project_id, adopt_stack_data=None, nested_depth=1)

    @mock.patch('heat.engine.environment.get_child_environment')
    @mock.patch.object(stack_resource.parser, 'Stack')
    def test_preview_with_implemented_dict_child_resource(self, mock_stack_class, mock_env_class):
        nested_stack = mock.Mock()
        mock_stack_class.return_value = nested_stack
        nested_stack.preview_resources.return_value = 'preview_nested_stack'
        mock_env_class.return_value = 'environment'
        template_dict = template_format.parse(param_template)
        parent_t = self.parent_stack.t
        resource_defns = parent_t.resource_definitions(self.parent_stack)
        parent_resource = MyImplementedStackResource('test', resource_defns[self.ws_resname], self.parent_stack)
        params = {'KeyName': 'test'}
        parent_resource.set_template(template_dict, params)
        validation_mock = mock.Mock(return_value=None)
        parent_resource._validate_nested_resources = validation_mock
        result = parent_resource.preview()
        mock_env_class.assert_called_once_with(self.parent_stack.env, params, child_resource_name='test', item_to_remove=None)
        self.assertEqual('preview_nested_stack', result)
        mock_stack_class.assert_called_once_with(mock.ANY, 'test_stack-test', mock.ANY, timeout_mins=None, disable_rollback=True, parent_resource=parent_resource.name, owner_id=self.parent_stack.id, user_creds_id=self.parent_stack.user_creds_id, stack_user_project_id=self.parent_stack.stack_user_project_id, adopt_stack_data=None, nested_depth=1)

    def test_preview_propagates_files(self):
        self.parent_stack.t.files['foo'] = 'bar'
        tmpl = self.parent_stack.t.t
        self.parent_resource.child_template = mock.Mock(return_value=tmpl)
        self.parent_resource.child_params = mock.Mock(return_value={})
        self.parent_resource.preview()
        self.stack = self.parent_resource.nested()
        self.assertEqual({'foo': 'bar'}, self.stack.t.files.files)

    def test_preview_validates_nested_resources(self):
        parent_t = self.parent_stack.t
        resource_defns = parent_t.resource_definitions(self.parent_stack)
        stk_resource = MyImplementedStackResource('test', resource_defns[self.ws_resname], self.parent_stack)
        stk_resource.child_params = mock.Mock(return_value={})
        stk_resource.child_template = mock.Mock(return_value=templatem.Template(self.simple_template, stk_resource.child_params))
        exc = exception.RequestLimitExceeded(message='Validation Failed')
        validation_mock = mock.Mock(side_effect=exc)
        stk_resource._validate_nested_resources = validation_mock
        self.assertRaises(exception.RequestLimitExceeded, stk_resource.preview)

    def test_parent_stack_existing_of_nested_stack(self):
        parent_t = self.parent_stack.t
        resource_defns = parent_t.resource_definitions(self.parent_stack)
        stk_resource = MyImplementedStackResource('test', resource_defns[self.ws_resname], self.parent_stack)
        stk_resource.child_params = mock.Mock(return_value={})
        stk_resource.child_template = mock.Mock(return_value=templatem.Template(self.simple_template, stk_resource.child_params))
        stk_resource._validate_nested_resources = mock.Mock()
        nest_stack = stk_resource._parse_nested_stack('test_nest_stack', stk_resource.child_template(), stk_resource.child_params())
        self.assertEqual(self.parent_stack, nest_stack.parent_resource._stack())

    def test_preview_dict_validates_nested_resources(self):
        parent_t = self.parent_stack.t
        resource_defns = parent_t.resource_definitions(self.parent_stack)
        stk_resource = MyImplementedStackResource('test', resource_defns[self.ws_resname], self.parent_stack)
        stk_resource.child_params = mock.Mock(return_value={})
        stk_resource.child_template = mock.Mock(return_value=self.simple_template)
        exc = exception.RequestLimitExceeded(message='Validation Failed')
        validation_mock = mock.Mock(side_effect=exc)
        stk_resource._validate_nested_resources = validation_mock
        self.assertRaises(exception.RequestLimitExceeded, stk_resource.preview)

    @mock.patch.object(stack_resource.parser, 'Stack')
    def test_preview_doesnt_validate_nested_stack(self, mock_stack_class):
        nested_stack = mock.Mock()
        mock_stack_class.return_value = nested_stack
        tmpl = self.parent_stack.t.t
        self.parent_resource.child_template = mock.Mock(return_value=tmpl)
        self.parent_resource.child_params = mock.Mock(return_value={})
        self.parent_resource.preview()
        self.assertFalse(nested_stack.validate.called)
        self.assertTrue(nested_stack.preview_resources.called)

    def test_validate_error_reference(self):
        stack_name = 'validate_error_reference'
        tmpl = template_format.parse(main_template)
        files = {'file://tmp/nested.yaml': my_wrong_nested_template}
        stack = parser.Stack(utils.dummy_context(), stack_name, templatem.Template(tmpl, files=files))
        rsrc = stack['volume_server']
        raise_exc_msg = 'InvalidTemplateReference: resources.volume_server<file://tmp/nested.yaml>: The specified reference "instance" (in volume_attachment.Properties.instance_uuid) is incorrect.'
        exc = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
        self.assertEqual(raise_exc_msg, str(exc))

    def _test_validate_unknown_resource_type(self, stack_name, tmpl, resource_name):
        raise_exc_msg = 'The Resource Type (idontexist) could not be found.'
        stack = parser.Stack(utils.dummy_context(), stack_name, tmpl)
        rsrc = stack[resource_name]
        exc = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
        self.assertIn(raise_exc_msg, str(exc))

    def test_validate_resource_group(self):
        stack_name = 'validate_resource_group_template'
        t = template_format.parse(resource_group_template)
        tmpl = templatem.Template(t)
        self._test_validate_unknown_resource_type(stack_name, tmpl, 'my_resource_group')
        res_prop = t['resources']['my_resource_group']['properties']
        res_prop['resource_def']['type'] = 'nova_server.yaml'
        files = {'nova_server.yaml': nova_server_template}
        tmpl = templatem.Template(t, files=files)
        self._test_validate_unknown_resource_type(stack_name, tmpl, 'my_resource_group')

    def test_validate_heat_autoscaling_group(self):
        stack_name = 'validate_heat_autoscaling_group_template'
        t = template_format.parse(heat_autoscaling_group_template)
        tmpl = templatem.Template(t)
        self._test_validate_unknown_resource_type(stack_name, tmpl, 'my_autoscaling_group')
        res_prop = t['resources']['my_autoscaling_group']['properties']
        res_prop['resource']['type'] = 'nova_server.yaml'
        files = {'nova_server.yaml': nova_server_template}
        tmpl = templatem.Template(t, files=files)
        self._test_validate_unknown_resource_type(stack_name, tmpl, 'my_autoscaling_group')

    def test_get_attribute_autoscaling(self):
        t = template_format.parse(heat_autoscaling_group_template)
        tmpl = templatem.Template(t)
        stack = parser.Stack(utils.dummy_context(), 'test_att', tmpl)
        rsrc = stack['my_autoscaling_group']
        self.assertEqual(0, rsrc.FnGetAtt(rsrc.CURRENT_SIZE))

    def test_get_attribute_autoscaling_convg(self):
        t = template_format.parse(heat_autoscaling_group_template)
        tmpl = templatem.Template(t)
        cache_data = {'my_autoscaling_group': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'attrs': {'current_size': 4}})}
        stack = parser.Stack(utils.dummy_context(), 'test_att', tmpl, cache_data=cache_data)
        rsrc = stack.defn['my_autoscaling_group']
        self.assertEqual(4, rsrc.FnGetAtt('current_size'))

    def test__validate_nested_resources_checks_num_of_resources(self):
        stack_resource.cfg.CONF.set_override('max_resources_per_stack', 2)
        tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'r': {'Type': 'OS::Heat::None'}}}
        template = stack_resource.template.Template(tmpl)
        root_resources = mock.Mock(return_value=2)
        self.parent_resource.stack.total_resources = root_resources
        self.assertRaises(exception.RequestLimitExceeded, self.parent_resource._validate_nested_resources, template)

    def test_load_nested_ok(self):
        self.parent_resource._nested = None
        self.parent_resource.resource_id = 319
        mock_load = self.patchobject(parser.Stack, 'load', return_value='s')
        self.parent_resource.nested()
        mock_load.assert_called_once_with(self.parent_resource.context, self.parent_resource.resource_id)

    def test_load_nested_non_exist(self):
        self.parent_resource._nested = None
        self.parent_resource.resource_id = '90-8'
        mock_load = self.patchobject(parser.Stack, 'load', side_effect=[exception.NotFound])
        self.assertIsNone(self.parent_resource.nested())
        mock_load.assert_called_once_with(self.parent_resource.context, self.parent_resource.resource_id)

    def test_load_nested_cached(self):
        self.parent_resource._nested = 'gotthis'
        self.assertEqual('gotthis', self.parent_resource.nested())

    def test_delete_nested_none_nested_stack(self):
        self.parent_resource._nested = None
        self.assertIsNone(self.parent_resource.delete_nested())

    def test_delete_nested_not_found_nested_stack(self):
        self.parent_resource.resource_id = 'fake_id'
        rpcc = mock.MagicMock()
        self.parent_resource.rpc_client = rpcc

        @contextlib.contextmanager
        def exc_filter(*args):
            try:
                yield
            except exception.EntityNotFound:
                pass
        rpcc.return_value.ignore_error_by_name.side_effect = exc_filter
        rpcc.return_value.delete_stack = mock.Mock(side_effect=exception.EntityNotFound('Stack', 'nested'))
        status = ('CREATE', 'COMPLETE', '', 'now_time')
        with mock.patch.object(stack_object.Stack, 'get_status', return_value=status):
            self.assertIsNone(self.parent_resource.delete_nested())
        rpcc.return_value.delete_stack.assert_called_once_with(self.parent_resource.context, mock.ANY, cast=False)

    def test_need_update_for_nested_resource(self):
        """Test the resource with nested stack should need update.

        The resource in Create or Update state and has nested stack, should
        need update.
        """
        self.parent_resource.action = self.parent_resource.CREATE
        self.parent_resource._rpc_client = mock.MagicMock()
        self.parent_resource._rpc_client.show_stack.return_value = [{'stack_action': self.parent_resource.CREATE, 'stack_status': self.parent_resource.COMPLETE}]
        need_update = self.parent_resource._needs_update(self.parent_resource.t, self.parent_resource.t, self.parent_resource.properties, self.parent_resource.properties, self.parent_resource)
        self.assertTrue(need_update)

    def test_need_update_in_failed_state_for_nested_resource(self):
        """Test the resource with no nested stack should need replacement.

        The resource in failed state and has no nested stack,
        should need update with UpdateReplace.
        """
        self.parent_resource.state_set(self.parent_resource.INIT, self.parent_resource.FAILED)
        self.parent_resource._nested = None
        self.assertRaises(resource.UpdateReplace, self.parent_resource._needs_update, self.parent_resource.t, self.parent_resource.t, self.parent_resource.properties, self.parent_resource.properties, self.parent_resource)

    def test_need_update_in_init_complete_state_for_nested_resource(self):
        """Test the resource with no nested stack should need replacement.

        The resource in failed state and has no nested stack,
        should need update with UpdateReplace.
        """
        self.parent_resource.state_set(self.parent_resource.INIT, self.parent_resource.COMPLETE)
        self.parent_resource._nested = None
        self.assertRaises(resource.UpdateReplace, self.parent_resource._needs_update, self.parent_resource.t, self.parent_resource.t, self.parent_resource.properties, self.parent_resource.properties, self.parent_resource)

    def test_need_update_in_check_failed_state_after_stack_check(self):
        self.parent_resource.resource_id = 'fake_id'
        self.parent_resource.state_set(self.parent_resource.CHECK, self.parent_resource.FAILED)
        self.nested = mock.MagicMock()
        self.nested.name = 'nested-stack'
        self.parent_resource.nested = mock.MagicMock(return_value=self.nested)
        self.parent_resource._nested = self.nested
        self.parent_resource._rpc_client = mock.MagicMock()
        self.parent_resource._rpc_client.show_stack.return_value = [{'stack_action': self.parent_resource.CHECK, 'stack_status': self.parent_resource.FAILED}]
        self.assertTrue(self.parent_resource._needs_update(self.parent_resource.t, self.parent_resource.t, self.parent_resource.properties, self.parent_resource.properties, self.parent_resource))

    def test_need_update_check_failed_state_after_mark_unhealthy(self):
        self.parent_resource.resource_id = 'fake_id'
        self.parent_resource.state_set(self.parent_resource.CHECK, self.parent_resource.FAILED)
        self.nested = mock.MagicMock()
        self.nested.name = 'nested-stack'
        self.parent_resource.nested = mock.MagicMock(return_value=self.nested)
        self.parent_resource._nested = self.nested
        self.parent_resource._rpc_client = mock.MagicMock()
        self.parent_resource._rpc_client.show_stack.return_value = [{'stack_action': self.parent_resource.CREATE, 'stack_status': self.parent_resource.COMPLETE}]
        self.assertRaises(resource.UpdateReplace, self.parent_resource._needs_update, self.parent_resource.t, self.parent_resource.t, self.parent_resource.properties, self.parent_resource.properties, self.parent_resource)