import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_multipath_device_path')
def _test_connect_volume_multipath(self, get_device_info_mock, get_scsi_wwn_mock, get_fc_hbas_info_mock, get_fc_hbas_mock, realpath_mock, exists_mock, wait_for_rw_mock, find_mp_dev_mock, access_mode, should_wait_for_rw, find_mp_device_path_mock):
    self.connector.use_multipath = True
    get_fc_hbas_mock.side_effect = self.fake_get_fc_hbas
    get_fc_hbas_info_mock.side_effect = self.fake_get_fc_hbas_info
    wwn = '1234567890'
    multipath_devname = '/dev/md-1'
    devices = {'device': multipath_devname, 'id': wwn, 'devices': [{'device': '/dev/sdb', 'address': '1:0:0:1', 'host': 1, 'channel': 0, 'id': 0, 'lun': 1}, {'device': '/dev/sdc', 'address': '1:0:0:2', 'host': 1, 'channel': 0, 'id': 0, 'lun': 1}]}
    get_device_info_mock.side_effect = devices['devices']
    get_scsi_wwn_mock.return_value = wwn
    location = '10.0.2.15:3260'
    name = 'volume-00000001'
    vol = {'id': 1, 'name': name}
    initiator_wwn = ['1234567890123456', '1234567890123457']
    find_mp_device_path_mock.return_value = '/dev/mapper/mpatha'
    find_mp_dev_mock.return_value = {'device': 'dm-3', 'id': wwn, 'name': 'mpatha'}
    connection_info = self.fibrechan_connection(vol, location, initiator_wwn)
    connection_info['data']['access_mode'] = access_mode
    self.connector.connect_volume(connection_info['data'])
    self.assertEqual(should_wait_for_rw, wait_for_rw_mock.called)
    self.connector.disconnect_volume(connection_info['data'], devices['devices'][0])
    expected_commands = ['multipath -f ' + find_mp_device_path_mock.return_value, 'tee -a /sys/block/sdb/device/delete', 'tee -a /sys/block/sdc/device/delete']
    self.assertEqual(expected_commands, self.cmds)
    return connection_info