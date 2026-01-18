from unittest import mock
from heat.engine import attributes
from heat.engine import resources
from heat.engine import support
from heat.tests import common
class AttributesTest(common.HeatTestCase):
    """Test the Attributes class."""

    def setUp(self):
        super(AttributesTest, self).setUp()
        self.resolver = mock.MagicMock()
        self.attributes_schema = {'test1': attributes.Schema('Test attrib 1'), 'test2': attributes.Schema('Test attrib 2'), 'test3': attributes.Schema('Test attrib 3', cache_mode=attributes.Schema.CACHE_NONE)}

    def test_get_attribute(self):
        """Test that we get the attribute values we expect."""
        self.resolver.return_value = 'value1'
        attribs = attributes.Attributes('test resource', self.attributes_schema, self.resolver)
        self.assertEqual('value1', attribs['test1'])
        self.resolver.assert_called_once_with('test1')

    def test_attributes_representation(self):
        """Test that attributes are displayed correct."""
        self.resolver.return_value = 'value1'
        attribs = attributes.Attributes('test resource', self.attributes_schema, self.resolver)
        msg = 'Attributes for test resource:\n\tvalue1\n\tvalue1\n\tvalue1'
        self.assertEqual(msg, str(attribs))
        calls = [mock.call('test1'), mock.call('test2'), mock.call('test3')]
        self.resolver.assert_has_calls(calls, any_order=True)

    def test_get_attribute_none(self):
        """Test that we get the attribute values we expect."""
        self.resolver.return_value = None
        attribs = attributes.Attributes('test resource', self.attributes_schema, self.resolver)
        self.assertIsNone(attribs['test1'])
        self.resolver.assert_called_once_with('test1')

    def test_get_attribute_nonexist(self):
        """Test that we get the attribute values we expect."""
        self.resolver.return_value = 'value1'
        attribs = attributes.Attributes('test resource', self.attributes_schema, self.resolver)
        self.assertRaises(KeyError, attribs.__getitem__, 'not there')
        self.assertFalse(self.resolver.called)

    def test_as_outputs(self):
        """Test that Output format works as expected."""
        expected = {'test1': {'Value': {'Fn::GetAtt': ['test_resource', 'test1']}, 'Description': 'Test attrib 1'}, 'test2': {'Value': {'Fn::GetAtt': ['test_resource', 'test2']}, 'Description': 'Test attrib 2'}, 'test3': {'Value': {'Fn::GetAtt': ['test_resource', 'test3']}, 'Description': 'Test attrib 3'}, 'OS::stack_id': {'Value': {'Ref': 'test_resource'}}}
        MyTestResourceClass = mock.MagicMock()
        MyTestResourceClass.attributes_schema = {'test1': attributes.Schema('Test attrib 1'), 'test2': attributes.Schema('Test attrib 2'), 'test3': attributes.Schema('Test attrib 3'), 'test4': attributes.Schema('Test attrib 4', support_status=support.SupportStatus(status=support.HIDDEN))}
        self.assertEqual(expected, attributes.Attributes.as_outputs('test_resource', MyTestResourceClass))

    def test_as_outputs_hot(self):
        """Test that Output format works as expected."""
        expected = {'test1': {'value': {'get_attr': ['test_resource', 'test1']}, 'description': 'Test attrib 1'}, 'test2': {'value': {'get_attr': ['test_resource', 'test2']}, 'description': 'Test attrib 2'}, 'test3': {'value': {'get_attr': ['test_resource', 'test3']}, 'description': 'Test attrib 3'}, 'OS::stack_id': {'value': {'get_resource': 'test_resource'}}}
        MyTestResourceClass = mock.MagicMock()
        MyTestResourceClass.attributes_schema = {'test1': attributes.Schema('Test attrib 1'), 'test2': attributes.Schema('Test attrib 2'), 'test3': attributes.Schema('Test attrib 3'), 'test4': attributes.Schema('Test attrib 4', support_status=support.SupportStatus(status=support.HIDDEN))}
        self.assertEqual(expected, attributes.Attributes.as_outputs('test_resource', MyTestResourceClass, 'hot'))

    def test_caching_local(self):
        self.resolver.side_effect = ['value1', 'value1 changed']
        attribs = attributes.Attributes('test resource', self.attributes_schema, self.resolver)
        self.assertEqual('value1', attribs['test1'])
        self.assertEqual('value1', attribs['test1'])
        attribs.reset_resolved_values()
        self.assertEqual('value1 changed', attribs['test1'])
        calls = [mock.call('test1'), mock.call('test1')]
        self.resolver.assert_has_calls(calls)

    def test_caching_none(self):
        self.resolver.side_effect = ['value3', 'value3 changed']
        attribs = attributes.Attributes('test resource', self.attributes_schema, self.resolver)
        self.assertEqual('value3', attribs['test3'])
        self.assertEqual('value3 changed', attribs['test3'])
        calls = [mock.call('test3'), mock.call('test3')]
        self.resolver.assert_has_calls(calls)