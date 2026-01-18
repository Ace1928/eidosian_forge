import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
class FormatTest(common.HeatTestCase):

    def setUp(self):
        super(FormatTest, self).setUp()
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'generic1': {'Type': 'GenericResourceType', 'Properties': {'k1': 'v1'}}, 'generic2': {'Type': 'GenericResourceType', 'DependsOn': 'generic1'}, 'generic3': {'Type': 'ResWithShowAttrType'}, 'generic4': {'Type': 'StackResourceType'}}})
        self.context = utils.dummy_context()
        self.stack = parser.Stack(self.context, 'test_stack', tmpl, stack_id=str(uuid.uuid4()))

    def _dummy_event(self, res_properties=None):
        resource = self.stack['generic1']
        ev_uuid = 'abc123yc-9f88-404d-a85b-531529456xyz'
        ev = event.Event(self.context, self.stack, 'CREATE', 'COMPLETE', 'state changed', 'z3455xyc-9f88-404d-a85b-5315293e67de', resource._rsrc_prop_data_id, resource._stored_properties_data, resource.name, resource.type(), uuid=ev_uuid)
        ev.store()
        return event_object.Event.get_all_by_stack(self.context, self.stack.id, filters={'uuid': ev_uuid})[0]

    def test_format_stack_resource(self):
        self.stack.created_time = datetime(2015, 8, 3, 17, 5, 1)
        self.stack.updated_time = datetime(2015, 8, 3, 17, 6, 2)
        res = self.stack['generic1']
        resource_keys = set((rpc_api.RES_CREATION_TIME, rpc_api.RES_UPDATED_TIME, rpc_api.RES_NAME, rpc_api.RES_PHYSICAL_ID, rpc_api.RES_ACTION, rpc_api.RES_STATUS, rpc_api.RES_STATUS_DATA, rpc_api.RES_TYPE, rpc_api.RES_ID, rpc_api.RES_STACK_ID, rpc_api.RES_STACK_NAME, rpc_api.RES_REQUIRED_BY))
        resource_details_keys = resource_keys.union(set((rpc_api.RES_DESCRIPTION, rpc_api.RES_METADATA, rpc_api.RES_ATTRIBUTES)))
        formatted = api.format_stack_resource(res, True)
        self.assertEqual(resource_details_keys, set(formatted.keys()))
        formatted = api.format_stack_resource(res, False)
        self.assertEqual(resource_keys, set(formatted.keys()))
        self.assertEqual(heat_timeutils.isotime(self.stack.created_time), formatted[rpc_api.RES_CREATION_TIME])
        self.assertEqual(heat_timeutils.isotime(self.stack.updated_time), formatted[rpc_api.RES_UPDATED_TIME])
        self.assertEqual(res.INIT, formatted[rpc_api.RES_ACTION])

    def test_format_stack_resource_no_attrs(self):
        res = self.stack['generic1']
        formatted = api.format_stack_resource(res, True, with_attr=False)
        self.assertNotIn(rpc_api.RES_ATTRIBUTES, formatted)
        self.assertIn(rpc_api.RES_METADATA, formatted)

    def test_format_stack_resource_has_been_deleted(self):
        self.stack.state_set(self.stack.DELETE, self.stack.COMPLETE, 'test_delete')
        res = self.stack['generic1']
        formatted = api.format_stack_resource(res, False)
        self.assertEqual(res.DELETE, formatted[rpc_api.RES_ACTION])

    def test_format_stack_resource_has_been_rollback(self):
        self.stack.state_set(self.stack.ROLLBACK, self.stack.COMPLETE, 'test_rollback')
        res = self.stack['generic1']
        formatted = api.format_stack_resource(res, False)
        self.assertEqual(res.ROLLBACK, formatted[rpc_api.RES_ACTION])

    @mock.patch.object(api, 'format_resource_properties')
    def test_format_stack_resource_with_props(self, mock_format_props):
        mock_format_props.return_value = 'formatted_res_props'
        res = self.stack['generic1']
        formatted = api.format_stack_resource(res, True, with_props=True)
        formatted_props = formatted[rpc_api.RES_PROPERTIES]
        self.assertEqual('formatted_res_props', formatted_props)

    @mock.patch.object(api, 'format_resource_attributes')
    def test_format_stack_resource_with_attributes(self, mock_format_attrs):
        mock_format_attrs.return_value = 'formatted_resource_attrs'
        res = self.stack['generic1']
        formatted = api.format_stack_resource(res, True, with_attr=['a', 'b'])
        formatted_attrs = formatted[rpc_api.RES_ATTRIBUTES]
        self.assertEqual('formatted_resource_attrs', formatted_attrs)

    def test_format_resource_attributes(self):
        res = self.stack['generic1']
        formatted_attributes = api.format_resource_attributes(res)
        expected = {'foo': 'generic1', 'Foo': 'generic1'}
        self.assertEqual(expected, formatted_attributes)

    def test_format_resource_attributes_show_attribute(self):
        res = self.stack['generic3']
        res.resource_id = 'generic3_id'
        formatted_attributes = api.format_resource_attributes(res)
        self.assertEqual(3, len(formatted_attributes))
        self.assertIn('foo', formatted_attributes)
        self.assertIn('Foo', formatted_attributes)
        self.assertIn('Another', formatted_attributes)

    def test_format_resource_attributes_show_attribute_with_attr(self):
        res = self.stack['generic3']
        res.resource_id = 'generic3_id'
        formatted_attributes = api.format_resource_attributes(res, with_attr=['c'])
        self.assertEqual(4, len(formatted_attributes))
        self.assertIn('foo', formatted_attributes)
        self.assertIn('Foo', formatted_attributes)
        self.assertIn('Another', formatted_attributes)
        self.assertIn('c', formatted_attributes)

    def _get_formatted_resource_properties(self, res_name):
        tmpl = template.Template(template_format.parse('\n            heat_template_version: 2013-05-23\n            resources:\n              resource1:\n                type: ResWithComplexPropsAndAttrs\n              resource2:\n                type: ResWithComplexPropsAndAttrs\n                properties:\n                  a_string: foobar\n              resource3:\n                type: ResWithComplexPropsAndAttrs\n                properties:\n                  a_string: { get_attr: [ resource2, string] }\n        '))
        stack = parser.Stack(utils.dummy_context(), 'test_stack_for_preview', tmpl, stack_id=str(uuid.uuid4()))
        res = stack[res_name]
        return api.format_resource_properties(res)

    def test_format_resource_properties_empty(self):
        props = self._get_formatted_resource_properties('resource1')
        self.assertIsNone(props['a_string'])
        self.assertIsNone(props['a_list'])
        self.assertIsNone(props['a_map'])

    def test_format_resource_properties_direct_props(self):
        props = self._get_formatted_resource_properties('resource2')
        self.assertEqual('foobar', props['a_string'])

    def test_format_resource_properties_get_attr(self):
        props = self._get_formatted_resource_properties('resource3')
        self.assertEqual('', props['a_string'])

    def test_format_stack_resource_with_nested_stack(self):
        res = self.stack['generic4']
        nested_id = {'foo': 'bar'}
        res.has_nested = mock.Mock()
        res.has_nested.return_value = True
        res.nested_identifier = mock.Mock()
        res.nested_identifier.return_value = nested_id
        formatted = api.format_stack_resource(res, False)
        self.assertEqual(nested_id, formatted[rpc_api.RES_NESTED_STACK_ID])

    def test_format_stack_resource_with_nested_stack_none(self):
        res = self.stack['generic4']
        resource_keys = set((rpc_api.RES_CREATION_TIME, rpc_api.RES_UPDATED_TIME, rpc_api.RES_NAME, rpc_api.RES_PHYSICAL_ID, rpc_api.RES_ACTION, rpc_api.RES_STATUS, rpc_api.RES_STATUS_DATA, rpc_api.RES_TYPE, rpc_api.RES_ID, rpc_api.RES_STACK_ID, rpc_api.RES_STACK_NAME, rpc_api.RES_REQUIRED_BY))
        formatted = api.format_stack_resource(res, False)
        self.assertEqual(resource_keys, set(formatted.keys()))

    def test_format_stack_resource_with_nested_stack_not_found(self):
        res = self.stack['generic4']
        self.patchobject(parser.Stack, 'load', side_effect=exception.NotFound())
        resource_keys = set((rpc_api.RES_CREATION_TIME, rpc_api.RES_UPDATED_TIME, rpc_api.RES_NAME, rpc_api.RES_PHYSICAL_ID, rpc_api.RES_ACTION, rpc_api.RES_STATUS, rpc_api.RES_STATUS_DATA, rpc_api.RES_TYPE, rpc_api.RES_ID, rpc_api.RES_STACK_ID, rpc_api.RES_STACK_NAME, rpc_api.RES_REQUIRED_BY))
        formatted = api.format_stack_resource(res, False)
        self.assertEqual(resource_keys, set(formatted.keys()))

    def test_format_stack_resource_with_nested_stack_empty(self):
        res = self.stack['generic4']
        nested_id = {'foo': 'bar'}
        res.has_nested = mock.Mock()
        res.has_nested.return_value = True
        res.nested_identifier = mock.Mock()
        res.nested_identifier.return_value = nested_id
        formatted = api.format_stack_resource(res, False)
        self.assertEqual(nested_id, formatted[rpc_api.RES_NESTED_STACK_ID])

    def test_format_stack_resource_required_by(self):
        res1 = api.format_stack_resource(self.stack['generic1'])
        res2 = api.format_stack_resource(self.stack['generic2'])
        self.assertEqual(['generic2'], res1['required_by'])
        self.assertEqual([], res2['required_by'])

    def test_format_stack_resource_with_parent_stack(self):
        res = self.stack['generic1']
        res.stack.defn._parent_info = parent_rsrc.ParentResourceProxy(self.stack.context, 'foobar', None)
        formatted = api.format_stack_resource(res, False)
        self.assertEqual('foobar', formatted[rpc_api.RES_PARENT_RESOURCE])

    def test_format_event_identifier_uuid(self):
        event = self._dummy_event()
        event_keys = set((rpc_api.EVENT_ID, rpc_api.EVENT_STACK_ID, rpc_api.EVENT_STACK_NAME, rpc_api.EVENT_TIMESTAMP, rpc_api.EVENT_RES_NAME, rpc_api.EVENT_RES_PHYSICAL_ID, rpc_api.EVENT_RES_ACTION, rpc_api.EVENT_RES_STATUS, rpc_api.EVENT_RES_STATUS_DATA, rpc_api.EVENT_RES_TYPE, rpc_api.EVENT_RES_PROPERTIES))
        formatted = api.format_event(event, self.stack.identifier())
        self.assertEqual(event_keys, set(formatted.keys()))
        event_id_formatted = formatted[rpc_api.EVENT_ID]
        self.assertEqual({'path': '/resources/generic1/events/%s' % event.uuid, 'stack_id': self.stack.id, 'stack_name': 'test_stack', 'tenant': 'test_tenant_id'}, event_id_formatted)

    def test_format_event_prop_data(self):
        resource = self.stack['generic1']
        resource._update_stored_properties()
        resource.store()
        event = self._dummy_event(res_properties=resource._stored_properties_data)
        formatted = api.format_event(event, self.stack.identifier(), include_rsrc_prop_data=True)
        self.assertEqual({'k1': 'v1'}, formatted[rpc_api.EVENT_RES_PROPERTIES])

    def test_format_event_legacy_prop_data(self):
        event = self._dummy_event(res_properties=None)
        with db_api.context_manager.writer.using(self.stack.context):
            db_obj = self.stack.context.session.query(models.Event).filter_by(id=event.id).first()
            db_obj.update({'resource_properties': {'legacy_k1': 'legacy_v1'}})
            db_obj.save(self.stack.context.session)
        event_legacy = event_object.Event.get_all_by_stack(self.context, self.stack.id)[0]
        formatted = api.format_event(event_legacy, self.stack.identifier())
        self.assertEqual({'legacy_k1': 'legacy_v1'}, formatted[rpc_api.EVENT_RES_PROPERTIES])

    def test_format_event_empty_prop_data(self):
        event = self._dummy_event(res_properties=None)
        formatted = api.format_event(event, self.stack.identifier())
        self.assertEqual({}, formatted[rpc_api.EVENT_RES_PROPERTIES])

    @mock.patch.object(api, 'format_stack_resource')
    def test_format_stack_preview(self, mock_fmt_resource):

        def mock_format_resources(res, **kwargs):
            return 'fmt%s' % res
        mock_fmt_resource.side_effect = mock_format_resources
        resources = [1, [2, [3]]]
        self.stack.preview_resources = mock.Mock(return_value=resources)
        stack = api.format_stack_preview(self.stack)
        self.assertIsInstance(stack, dict)
        self.assertIsNone(stack.get('status'))
        self.assertIsNone(stack.get('action'))
        self.assertIsNone(stack.get('status_reason'))
        self.assertEqual('test_stack', stack['stack_name'])
        self.assertIn('resources', stack)
        resources = list(stack['resources'])
        self.assertEqual('fmt1', resources[0])
        resources = list(resources[1])
        self.assertEqual('fmt2', resources[0])
        resources = list(resources[1])
        self.assertEqual('fmt3', resources[0])
        kwargs = mock_fmt_resource.call_args[1]
        self.assertTrue(kwargs['with_props'])

    def test_format_stack(self):
        self.stack.created_time = datetime(1970, 1, 1)
        info = api.format_stack(self.stack)
        aws_id = 'arn:openstack:heat::test_tenant_id:stacks/test_stack/' + self.stack.id
        expected_stack_info = {'capabilities': [], 'creation_time': '1970-01-01T00:00:00Z', 'deletion_time': None, 'description': 'No description', 'disable_rollback': True, 'notification_topics': [], 'stack_action': 'CREATE', 'stack_name': 'test_stack', 'stack_owner': 'test_username', 'stack_status': 'IN_PROGRESS', 'stack_status_reason': '', 'stack_user_project_id': None, 'outputs': [], 'template_description': 'No description', 'timeout_mins': None, 'tags': [], 'parameters': {'AWS::Region': 'ap-southeast-1', 'AWS::StackId': aws_id, 'AWS::StackName': 'test_stack'}, 'stack_identity': {'path': '', 'stack_id': self.stack.id, 'stack_name': 'test_stack', 'tenant': 'test_tenant_id'}, 'updated_time': None, 'parent': None}
        self.assertEqual(expected_stack_info, info)

    def test_format_stack_created_time(self):
        self.stack.created_time = None
        info = api.format_stack(self.stack)
        self.assertIsNotNone(info['creation_time'])

    def test_format_stack_updated_time(self):
        self.stack.updated_time = None
        info = api.format_stack(self.stack)
        self.assertIsNone(info['updated_time'])
        self.stack.updated_time = datetime(1970, 1, 1)
        info = api.format_stack(self.stack)
        self.assertEqual('1970-01-01T00:00:00Z', info['updated_time'])

    @mock.patch.object(api, 'format_stack_outputs')
    def test_format_stack_adds_outputs(self, mock_fmt_outputs):
        mock_fmt_outputs.return_value = 'foobar'
        self.stack.action = 'CREATE'
        self.stack.status = 'COMPLETE'
        info = api.format_stack(self.stack)
        self.assertEqual('foobar', info[rpc_api.STACK_OUTPUTS])

    @mock.patch.object(api, 'format_stack_outputs')
    def test_format_stack_without_resolving_outputs(self, mock_fmt_outputs):
        mock_fmt_outputs.return_value = 'foobar'
        self.stack.action = 'CREATE'
        self.stack.status = 'COMPLETE'
        info = api.format_stack(self.stack, resolve_outputs=False)
        self.assertIsNone(info.get(rpc_api.STACK_OUTPUTS))

    def test_format_stack_outputs(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'generic': {'Type': 'GenericResourceType'}}, 'Outputs': {'correct_output': {'Description': 'Good output', 'Value': {'Fn::GetAtt': ['generic', 'Foo']}}, 'incorrect_output': {'Value': {'Fn::GetAtt': ['generic', 'Bar']}}}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl, stack_id=str(uuid.uuid4()))
        stack.action = 'CREATE'
        stack.status = 'COMPLETE'
        stack['generic'].action = 'CREATE'
        stack['generic'].status = 'COMPLETE'
        stack._update_all_resource_data(False, True)
        info = api.format_stack_outputs(stack.outputs, resolve_value=True)
        expected = [{'description': 'No description given', 'output_error': 'The Referenced Attribute (generic Bar) is incorrect.', 'output_key': 'incorrect_output', 'output_value': None}, {'description': 'Good output', 'output_key': 'correct_output', 'output_value': 'generic'}]
        self.assertEqual(expected, sorted(info, key=lambda k: k['output_key'], reverse=True))

    def test_format_stack_outputs_unresolved(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'generic': {'Type': 'GenericResourceType'}}, 'Outputs': {'correct_output': {'Description': 'Good output', 'Value': {'Fn::GetAtt': ['generic', 'Foo']}}, 'incorrect_output': {'Value': {'Fn::GetAtt': ['generic', 'Bar']}}}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl, stack_id=str(uuid.uuid4()))
        stack.action = 'CREATE'
        stack.status = 'COMPLETE'
        stack['generic'].action = 'CREATE'
        stack['generic'].status = 'COMPLETE'
        info = api.format_stack_outputs(stack.outputs)
        expected = [{'description': 'No description given', 'output_key': 'incorrect_output'}, {'description': 'Good output', 'output_key': 'correct_output'}]
        self.assertEqual(expected, sorted(info, key=lambda k: k['output_key'], reverse=True))

    def test_format_stack_params_csv(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'parameters': {'foo': {'type': 'comma_delimited_list', 'default': ['bar', 'baz']}}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl, stack_id=str(uuid.uuid4()))
        info = api.format_stack(stack)
        self.assertEqual('bar,baz', info['parameters']['foo'])

    def test_format_stack_params_json(self):
        tmpl = template.Template({'heat_template_version': '2013-05-23', 'parameters': {'foo': {'type': 'json', 'default': {'bar': 'baz'}}}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl, stack_id=str(uuid.uuid4()))
        info = api.format_stack(stack)
        self.assertEqual('{"bar": "baz"}', info['parameters']['foo'])