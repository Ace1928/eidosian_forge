from novaclient.tests.functional.v2.legacy import test_server_groups
class TestServerGroupClientV264(TestServerGroupClientV213):
    """Server groups v2.64 functional tests."""
    COMPUTE_API_VERSION = '2.64'
    expected_metadata = False
    expected_policy_rules = True

    def test_create_server_group(self):
        output = self.nova('server-group-create complex-anti-affinity-group anti-affinity --rule max_server_per_host=3')
        sg_id = self._get_column_value_from_single_row_table(output, 'Id')
        self.addCleanup(self.nova, 'server-group-delete %s' % sg_id)
        sg = self.nova('server-group-get %s' % sg_id)
        result = self._get_column_value_from_single_row_table(sg, 'Id')
        self.assertEqual(sg_id, result)
        self._get_column_value_from_single_row_table(sg, 'User Id')
        self._get_column_value_from_single_row_table(sg, 'Project Id')
        self.assertNotIn('Metadata', sg)
        self.assertEqual('anti-affinity', self._get_column_value_from_single_row_table(sg, 'Policy'))
        self.assertIn('max_server_per_host', self._get_column_value_from_single_row_table(sg, 'Rules'))