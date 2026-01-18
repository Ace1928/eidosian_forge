from openstack.tests.functional.shared_file_system import base
class AvailabilityZoneTest(base.BaseSharedFileSystemTest):
    min_microversion = '2.7'

    def test_availability_zones(self):
        azs = self.user_cloud.shared_file_system.availability_zones()
        self.assertGreater(len(list(azs)), 0)
        for az in azs:
            for attribute in ('id', 'name', 'created_at', 'updated_at'):
                self.assertTrue(hasattr(az, attribute))
                self.assertIsInstance(getattr(az, attribute), 'str')