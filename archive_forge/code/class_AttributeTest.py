from unittest import mock
from heat.engine import attributes
from heat.engine import resources
from heat.engine import support
from heat.tests import common
class AttributeTest(common.HeatTestCase):
    """Test the Attribute class."""

    def test_as_output(self):
        """Test that Attribute looks right when viewed as an Output."""
        expected = {'Value': {'Fn::GetAtt': ['test_resource', 'test1']}, 'Description': 'The first test attribute'}
        attr = attributes.Attribute('test1', attributes.Schema('The first test attribute'))
        self.assertEqual(expected, attr.as_output('test_resource'))

    def test_as_output_hot(self):
        """Test that Attribute looks right when viewed as an Output."""
        expected = {'value': {'get_attr': ['test_resource', 'test1']}, 'description': 'The first test attribute'}
        attr = attributes.Attribute('test1', attributes.Schema('The first test attribute'))
        self.assertEqual(expected, attr.as_output('test_resource', 'hot'))