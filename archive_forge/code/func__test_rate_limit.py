from openstack.block_storage.v2 import limits
from openstack.tests.unit import base
def _test_rate_limit(self, expected, actual):
    self.assertEqual(expected[0]['verb'], actual[0].verb)
    self.assertEqual(expected[0]['value'], actual[0].value)
    self.assertEqual(expected[0]['remaining'], actual[0].remaining)
    self.assertEqual(expected[0]['unit'], actual[0].unit)
    self.assertEqual(expected[0]['next-available'], actual[0].next_available)