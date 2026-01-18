from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
from boto.ec2.elb.attributes import LbAttributes
def _verify_attributes(self, attributes, attr_tests):
    """Verifies an LbAttributes object."""
    for attr, result in attr_tests:
        attr_result = attributes
        for sub_attr in attr.split('.'):
            attr_result = getattr(attr_result, sub_attr, None)
        self.assertEqual(attr_result, result)