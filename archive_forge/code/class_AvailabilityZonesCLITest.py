from manilaclient.tests.functional.osc import base
class AvailabilityZonesCLITest(base.OSCClientTestBase):

    def test_openstack_share_availability_zones_list(self):
        azs = self.listing_result('share', 'availability zone list')
        self.assertTableStruct(azs, ['Id', 'Name', 'Created At', 'Updated At'])
        self.assertTrue(len(azs) > 0)