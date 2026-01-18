from openstackclient.tests.functional import base
class AvailabilityZoneTests(base.TestCase):
    """Functional tests for availability zone."""

    def test_availability_zone_list(self):
        cmd_output = self.openstack('availability zone list', parse_output=True)
        zones = [x['Zone Name'] for x in cmd_output]
        self.assertIn('internal', zones)
        self.assertIn('nova', zones)