from openstackclient.tests.functional.volume.v2 import common
class VolumeServiceTests(common.BaseVolumeTests):
    """Functional tests for volume service."""

    def test_volume_service_list(self):
        cmd_output = self.openstack('volume service list', parse_output=True)
        services = list(set([x['Binary'] for x in cmd_output]))
        hosts = list(set([x['Host'] for x in cmd_output]))
        cmd_output = self.openstack('volume service list ' + '--service ' + services[0], parse_output=True)
        for x in cmd_output:
            self.assertEqual(services[0], x['Binary'])
        cmd_output = self.openstack('volume service list ' + '--host ' + hosts[0], parse_output=True)
        for x in cmd_output:
            self.assertIn(hosts[0], x['Host'])

    def test_volume_service_set(self):
        cmd_output = self.openstack('volume service list', parse_output=True)
        service_1 = cmd_output[0]['Binary']
        host_1 = cmd_output[0]['Host']
        raw_output = self.openstack('volume service set --enable ' + host_1 + ' ' + service_1)
        self.assertOutput('', raw_output)
        cmd_output = self.openstack('volume service list --long', parse_output=True)
        self.assertEqual('enabled', cmd_output[0]['Status'])
        self.assertIsNone(cmd_output[0]['Disabled Reason'])
        disable_reason = 'disable_reason'
        raw_output = self.openstack('volume service set --disable ' + '--disable-reason ' + disable_reason + ' ' + host_1 + ' ' + service_1)
        self.assertOutput('', raw_output)
        cmd_output = self.openstack('volume service list --long', parse_output=True)
        self.assertEqual('disabled', cmd_output[0]['Status'])
        self.assertEqual(disable_reason, cmd_output[0]['Disabled Reason'])