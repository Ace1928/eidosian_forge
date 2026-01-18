from openstack.tests.functional import base
class TestComputeQuotas(base.BaseFunctionalTest):

    def test_get_quotas(self):
        """Test quotas functionality"""
        self.user_cloud.get_compute_quotas(self.user_cloud.current_project_id)

    def test_set_quotas(self):
        """Test quotas functionality"""
        if not self.operator_cloud:
            self.skipTest('Operator cloud is required for this test')
        quotas = self.operator_cloud.get_compute_quotas('demo')
        cores = quotas['cores']
        self.operator_cloud.set_compute_quotas('demo', cores=cores + 1)
        self.assertEqual(cores + 1, self.operator_cloud.get_compute_quotas('demo')['cores'])
        self.operator_cloud.delete_compute_quotas('demo')
        self.assertEqual(cores, self.operator_cloud.get_compute_quotas('demo')['cores'])