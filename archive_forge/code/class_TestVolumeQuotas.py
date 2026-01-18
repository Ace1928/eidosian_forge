from openstack.tests.functional import base
class TestVolumeQuotas(base.BaseFunctionalTest):

    def setUp(self):
        super(TestVolumeQuotas, self).setUp()
        if not self.user_cloud.has_service('volume'):
            self.skipTest('volume service not supported by cloud')

    def test_get_quotas(self):
        """Test get quotas functionality"""
        self.user_cloud.get_volume_quotas(self.user_cloud.current_project_id)

    def test_set_quotas(self):
        """Test set quotas functionality"""
        if not self.operator_cloud:
            self.skipTest('Operator cloud is required for this test')
        quotas = self.operator_cloud.get_volume_quotas('demo')
        volumes = quotas['volumes']
        self.operator_cloud.set_volume_quotas('demo', volumes=volumes + 1)
        self.assertEqual(volumes + 1, self.operator_cloud.get_volume_quotas('demo')['volumes'])
        self.operator_cloud.delete_volume_quotas('demo')
        self.assertEqual(volumes, self.operator_cloud.get_volume_quotas('demo')['volumes'])