from tempest.lib import exceptions
from novaclient.tests.functional import base
def _verify_quota_class_show_output(self, output, expected_values):
    for quota_name in self._included_resources:
        self.assertIn(quota_name, expected_values)
        expected_value = expected_values[quota_name]
        actual_value = self._get_value_from_the_table(output, quota_name)
        self.assertEqual(expected_value, actual_value)
    for quota_name in self._excluded_resources:
        self.assertRaises(ValueError, self._get_value_from_the_table, output, quota_name)