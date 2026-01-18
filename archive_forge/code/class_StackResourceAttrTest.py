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
class StackResourceAttrTest(StackResourceBaseTest):

    def test_get_output_ok(self):
        self.parent_resource.nested_identifier = mock.Mock()
        self.parent_resource.nested_identifier.return_value = {'foo': 'bar'}
        self.parent_resource._rpc_client = mock.MagicMock()
        output = {'outputs': [{'output_key': 'key', 'output_value': 'value'}]}
        self.parent_resource._rpc_client.show_stack.return_value = [output]
        self.assertEqual('value', self.parent_resource.get_output('key'))

    def test_get_output_key_not_found(self):
        self.parent_resource.nested_identifier = mock.Mock()
        self.parent_resource.nested_identifier.return_value = {'foo': 'bar'}
        self.parent_resource._rpc_client = mock.MagicMock()
        output = {'outputs': []}
        self.parent_resource._rpc_client.show_stack.return_value = [output]
        self.assertRaises(exception.NotFound, self.parent_resource.get_output, 'key')

    def test_get_output_key_no_outputs_from_rpc(self):
        self.parent_resource.nested_identifier = mock.Mock()
        self.parent_resource.nested_identifier.return_value = {'foo': 'bar'}
        self.parent_resource._rpc_client = mock.MagicMock()
        output = {}
        self.parent_resource._rpc_client.show_stack.return_value = [output]
        self.assertRaises(exception.NotFound, self.parent_resource.get_output, 'key')

    def test_resolve_attribute_string(self):
        self.parent_resource.nested_identifier = mock.Mock()
        self.parent_resource.nested_identifier.return_value = {'foo': 'bar'}
        self.parent_resource._rpc_client = mock.MagicMock()
        output = {'outputs': [{'output_key': 'key', 'output_value': 'value'}]}
        self.parent_resource._rpc_client.show_stack.return_value = [output]
        self.assertEqual('value', self.parent_resource._resolve_attribute('key'))

    def test_resolve_attribute_dict(self):
        self.parent_resource.nested_identifier = mock.Mock()
        self.parent_resource.nested_identifier.return_value = {'foo': 'bar'}
        self.parent_resource._rpc_client = mock.MagicMock()
        output = {'outputs': [{'output_key': 'key', 'output_value': {'a': 1, 'b': 2}}]}
        self.parent_resource._rpc_client.show_stack.return_value = [output]
        self.assertEqual({'a': 1, 'b': 2}, self.parent_resource._resolve_attribute('key'))

    def test_resolve_attribute_list(self):
        self.parent_resource.nested_identifier = mock.Mock()
        self.parent_resource.nested_identifier.return_value = {'foo': 'bar'}
        self.parent_resource._rpc_client = mock.MagicMock()
        output = {'outputs': [{'output_key': 'key', 'output_value': [1, 2, 3]}]}
        self.parent_resource._rpc_client.show_stack.return_value = [output]
        self.assertEqual([1, 2, 3], self.parent_resource._resolve_attribute('key'))

    def test_validate_nested_stack(self):
        self.parent_resource.child_template = mock.Mock(return_value='foo')
        self.parent_resource.child_params = mock.Mock(return_value={})
        nested = mock.Mock()
        nested.validate.return_value = True
        mock_parse_nested = self.patchobject(stack_resource.StackResource, '_parse_nested_stack', return_value=nested)
        name = '%s-%s' % (self.parent_stack.name, self.parent_resource.name)
        self.parent_resource.validate_nested_stack()
        self.assertFalse(nested.strict_validate)
        mock_parse_nested.assert_called_once_with(name, 'foo', {})

    def test_validate_assertion_exception_rethrow(self):
        expected_message = 'Expected Assertion Error'
        self.parent_resource.child_template = mock.Mock(return_value='foo')
        self.parent_resource.child_params = mock.Mock(return_value={})
        mock_parse_nested = self.patchobject(stack_resource.StackResource, '_parse_nested_stack', side_effect=AssertionError(expected_message))
        name = '%s-%s' % (self.parent_stack.name, self.parent_resource.name)
        exc = self.assertRaises(AssertionError, self.parent_resource.validate_nested_stack)
        self.assertEqual(expected_message, str(exc))
        mock_parse_nested.assert_called_once_with(name, 'foo', {})