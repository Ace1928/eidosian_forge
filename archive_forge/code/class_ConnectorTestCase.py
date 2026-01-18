import platform
import sys
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_service import loopingcall
from os_brick import exception
from os_brick.initiator import connector
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fake
from os_brick.initiator.connectors import iscsi
from os_brick.initiator.connectors import nvmeof
from os_brick.initiator import linuxfc
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick import utils
class ConnectorTestCase(test_base.TestCase):

    def setUp(self):
        super(ConnectorTestCase, self).setUp()
        self.cmds = []
        self.mock_object(loopingcall, 'FixedIntervalLoopingCall', ZeroIntervalLoopingCall)

    def fake_execute(self, *cmd, **kwargs):
        self.cmds.append(' '.join(cmd))
        return ('', None)

    def fake_connection(self):
        return {'driver_volume_type': 'fake', 'data': {'volume_id': 'fake_volume_id', 'target_portal': 'fake_location', 'target_iqn': 'fake_iqn', 'target_lun': 1}}

    def test_connect_volume(self):
        self.connector = fake.FakeConnector(None)
        device_info = self.connector.connect_volume(self.fake_connection())
        self.assertIn('type', device_info)
        self.assertIn('path', device_info)

    def test_disconnect_volume(self):
        self.connector = fake.FakeConnector(None)

    def test_get_connector_properties(self):
        with mock.patch.object(priv_rootwrap, 'execute') as mock_exec:
            mock_exec.return_value = ('', '')
            multipath = True
            enforce_multipath = True
            props = base.BaseLinuxConnector.get_connector_properties('sudo', multipath=multipath, enforce_multipath=enforce_multipath)
            expected_props = {'multipath': True}
            self.assertEqual(expected_props, props)
            multipath = False
            enforce_multipath = True
            props = base.BaseLinuxConnector.get_connector_properties('sudo', multipath=multipath, enforce_multipath=enforce_multipath)
            expected_props = {'multipath': False}
            self.assertEqual(expected_props, props)
        with mock.patch.object(priv_rootwrap, 'execute', side_effect=putils.ProcessExecutionError):
            multipath = True
            enforce_multipath = True
            self.assertRaises(putils.ProcessExecutionError, base.BaseLinuxConnector.get_connector_properties, 'sudo', multipath=multipath, enforce_multipath=enforce_multipath)

    @mock.patch('sys.platform', 'win32')
    def test_get_connector_mapping_win32(self):
        mapping_win32 = connector.get_connector_mapping()
        self.assertIn('ISCSI', mapping_win32)
        self.assertIn('RBD', mapping_win32)
        self.assertNotIn('STORPOOL', mapping_win32)

    @mock.patch('os_brick.initiator.connector.platform.machine')
    def test_get_connector_mapping(self, mock_platform_machine):
        mock_platform_machine.return_value = 'x86_64'
        mapping_x86 = connector.get_connector_mapping()
        mock_platform_machine.return_value = 'ppc64le'
        mapping_ppc = connector.get_connector_mapping()
        self.assertNotEqual(mapping_x86, mapping_ppc)
        mock_platform_machine.return_value = 's390x'
        mapping_s390 = connector.get_connector_mapping()
        self.assertNotEqual(mapping_x86, mapping_s390)
        self.assertNotEqual(mapping_ppc, mapping_s390)

    def test_factory(self):
        obj = connector.InitiatorConnector.factory('iscsi', None)
        self.assertEqual('ISCSIConnector', obj.__class__.__name__)
        obj = connector.InitiatorConnector.factory('iscsi', None, arch='ppc64le')
        self.assertEqual('ISCSIConnector', obj.__class__.__name__)
        obj = connector.InitiatorConnector.factory('fibre_channel', None, arch='x86_64')
        self.assertEqual('FibreChannelConnector', obj.__class__.__name__)
        obj = connector.InitiatorConnector.factory('fibre_channel', None, arch='s390x')
        self.assertEqual('FibreChannelConnectorS390X', obj.__class__.__name__)
        obj = connector.InitiatorConnector.factory('nfs', None, nfs_mount_point_base='/mnt/test')
        self.assertEqual('RemoteFsConnector', obj.__class__.__name__)
        obj = connector.InitiatorConnector.factory('glusterfs', None, glusterfs_mount_point_base='/mnt/test', arch='x86_64')
        self.assertEqual('RemoteFsConnector', obj.__class__.__name__)
        obj = connector.InitiatorConnector.factory('scality', None, scality_mount_point_base='/mnt/test', arch='x86_64')
        self.assertEqual('RemoteFsConnector', obj.__class__.__name__)
        obj = connector.InitiatorConnector.factory('local', None)
        self.assertEqual('LocalConnector', obj.__class__.__name__)
        obj = connector.InitiatorConnector.factory('gpfs', None)
        self.assertEqual('GPFSConnector', obj.__class__.__name__)
        obj = connector.InitiatorConnector.factory('huaweisdshypervisor', None, arch='x86_64')
        self.assertEqual('HuaweiStorHyperConnector', obj.__class__.__name__)
        obj = connector.InitiatorConnector.factory('scaleio', None, arch='x86_64')
        self.assertEqual('ScaleIOConnector', obj.__class__.__name__)
        obj = connector.InitiatorConnector.factory('quobyte', None, quobyte_mount_point_base='/mnt/test', arch='x86_64')
        self.assertEqual('RemoteFsConnector', obj.__class__.__name__)
        self.assertRaises(exception.InvalidConnectorProtocol, connector.InitiatorConnector.factory, 'bogus', None)

    def test_check_valid_device_with_wrong_path(self):
        self.connector = fake.FakeConnector(None)
        self.connector._execute = lambda *args, **kwargs: ('', None)
        self.assertFalse(self.connector.check_valid_device('/d0v'))

    def test_check_valid_device(self):
        self.connector = fake.FakeConnector(None)
        self.connector._execute = lambda *args, **kwargs: ('', '')
        self.assertTrue(self.connector.check_valid_device('/dev'))

    def test_check_valid_device_with_cmd_error(self):

        def raise_except(*args, **kwargs):
            raise putils.ProcessExecutionError
        self.connector = fake.FakeConnector(None)
        with mock.patch.object(self.connector, '_execute', side_effect=putils.ProcessExecutionError):
            self.assertFalse(self.connector.check_valid_device('/dev'))