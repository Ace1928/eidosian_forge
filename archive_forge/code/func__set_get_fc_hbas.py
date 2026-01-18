import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
def _set_get_fc_hbas(self):
    pci_path = '/sys/devices/pci0000:20/0000:20:03.0/0000:21:00.'
    host0_pci = f'{pci_path}0/host0/fc_host/host0'
    host2_pci = f'{pci_path}1/host2/fc_host/host2'
    return_value = [{'ClassDevice': 'host0', 'ClassDevicepath': host0_pci, 'port_name': '0x50014380242b9750', 'node_name': '0x50014380242b9751', 'port_state': 'Online'}, {'ClassDevice': 'host2', 'ClassDevicepath': host2_pci, 'port_name': '0x50014380242b9752', 'node_name': '0x50014380242b9753', 'port_state': 'Online'}]
    mocked = self.mock_object(linuxfc.LinuxFibreChannel, 'get_fc_hbas', return_value=return_value)
    return mocked