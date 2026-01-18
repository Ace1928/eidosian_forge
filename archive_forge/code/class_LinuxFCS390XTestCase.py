import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
class LinuxFCS390XTestCase(LinuxFCTestCase):

    def setUp(self):
        super(LinuxFCS390XTestCase, self).setUp()
        self.cmds = []
        self.lfc = linuxfc.LinuxFibreChannelS390X(None, execute=self.fake_execute)

    @mock.patch.object(linuxfc.LinuxFibreChannel, 'get_fc_hbas')
    def test_get_fc_hbas_info(self, mock_hbas):
        host_pci = '/sys/devices/css0/0.0.02ea/0.0.3080/host0/fc_host/host0'
        mock_hbas.return_value = [{'ClassDevice': 'host0', 'ClassDevicepath': host_pci, 'port_name': '0xc05076ffe680a960', 'node_name': '0x1234567898765432', 'port_state': 'Online'}]
        hbas_info = self.lfc.get_fc_hbas_info()
        expected = [{'device_path': '/sys/devices/css0/0.0.02ea/0.0.3080/host0/fc_host/host0', 'host_device': 'host0', 'node_name': '1234567898765432', 'port_name': 'c05076ffe680a960'}]
        self.assertEqual(expected, hbas_info)

    @mock.patch.object(os.path, 'exists', return_value=False)
    def test_configure_scsi_device(self, mock_execute):
        device_number = '0.0.2319'
        target_wwn = '0x50014380242b9751'
        lun = 1
        self.lfc.configure_scsi_device(device_number, target_wwn, lun)
        expected_commands = ['tee -a /sys/bus/ccw/drivers/zfcp/0.0.2319/port_rescan', 'tee -a /sys/bus/ccw/drivers/zfcp/0.0.2319/0x50014380242b9751/unit_add']
        self.assertEqual(expected_commands, self.cmds)

    def test_deconfigure_scsi_device(self):
        device_number = '0.0.2319'
        target_wwn = '0x50014380242b9751'
        lun = 1
        self.lfc.deconfigure_scsi_device(device_number, target_wwn, lun)
        expected_commands = ['tee -a /sys/bus/ccw/drivers/zfcp/0.0.2319/0x50014380242b9751/unit_remove']
        self.assertEqual(expected_commands, self.cmds)