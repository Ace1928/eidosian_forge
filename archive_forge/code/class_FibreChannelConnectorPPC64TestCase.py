from unittest import mock
from os_brick.initiator.connectors import fibre_channel_ppc64
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
class FibreChannelConnectorPPC64TestCase(test_connector.ConnectorTestCase):

    def setUp(self):
        super(FibreChannelConnectorPPC64TestCase, self).setUp()
        self.connector = fibre_channel_ppc64.FibreChannelConnectorPPC64(None, execute=self.fake_execute, use_multipath=False)
        self.assertIsNotNone(self.connector)
        self.assertIsNotNone(self.connector._linuxfc)
        self.assertEqual(self.connector._linuxfc.__class__.__name__, 'LinuxFibreChannel')
        self.assertIsNotNone(self.connector._linuxscsi)

    @mock.patch.object(linuxscsi.LinuxSCSI, 'process_lun_id', return_value='2')
    def test_get_host_devices(self, mock_process_lun_id):
        lun = 2
        possible_devs = [(3, '0x5005076802232ade'), (3, '0x5005076802332ade')]
        devices = self.connector._get_host_devices(possible_devs, lun)
        self.assertEqual(2, len(devices))
        device_path = '/dev/disk/by-path/fc-0x5005076802332ade-lun-2'
        self.assertIn(device_path, devices)
        device_path = '/dev/disk/by-path/fc-0x5005076802232ade-lun-2'
        self.assertIn(device_path, devices)
        possible_devs = [(3, '0x5005076802232ade'), (3, '0x5005076802232ade')]
        devices = self.connector._get_host_devices(possible_devs, lun)
        self.assertEqual(1, len(devices))
        device_path = '/dev/disk/by-path/fc-0x5005076802232ade-lun-2'
        self.assertIn(device_path, devices)