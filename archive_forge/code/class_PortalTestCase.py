import builtins
import errno
import os.path
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import executor
from os_brick.initiator.connectors import nvmeof
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick.tests.initiator import test_connector
from os_brick import utils
@ddt.ddt
class PortalTestCase(test_base.TestCase):

    def setUp(self):
        self.conn_props_dict = {'target_nqn': 'nqn_value', 'vol_uuid': VOL_UUID, 'portals': [('portal1', 'port1', 'RoCEv2')]}
        self.conn_props = nvmeof.NVMeOFConnProps(self.conn_props_dict)
        self.target = self.conn_props.targets[0]
        self.portal = self.target.portals[0]
        super().setUp()

    @ddt.data(('RoCEv2', 'rdma'), ('rdma', 'rdma'), ('tcp', 'tcp'), ('TCP', 'tcp'), ('other', 'tcp'))
    @ddt.unpack
    def test_init(self, transport, expected_transport):
        """Init changes conn props transport into rdma or tcp."""
        portal = nvmeof.Portal(self.target, 'address', 'port', transport)
        self.assertEqual(self.target, portal.parent_target)
        self.assertEqual('address', portal.address)
        self.assertEqual('port', portal.port)
        self.assertIsNone(portal.controller)
        self.assertEqual(expected_transport, portal.transport)

    @ddt.data(('live', True), ('connecting', False), (None, False))
    @ddt.unpack
    @mock.patch.object(nvmeof.Portal, 'state', new_callable=mock.PropertyMock)
    def test_is_live(self, state, expected, mock_state):
        """Is live only returns True if the state is 'live'."""
        mock_state.return_value = state
        self.assertIs(expected, self.portal.is_live)
        mock_state.assert_called_once_with()

    @mock.patch.object(nvmeof, 'ctrl_property', return_value='10')
    def test_reconnect_delay(self, mock_property):
        """Reconnect delay returns an int."""
        self.portal.controller = 'nvme0'
        self.assertIs(10, self.portal.reconnect_delay)
        mock_property.assert_called_once_with('reconnect_delay', 'nvme0')

    @mock.patch.object(nvmeof, 'ctrl_property')
    def test_state(self, mock_property):
        """State uses sysfs to check the value."""
        self.portal.controller = 'nvme0'
        self.assertEqual(mock_property.return_value, self.portal.state)
        mock_property.assert_called_once_with('state', 'nvme0')

    @mock.patch.object(nvmeof, 'ctrl_property')
    def test_state_no_controller(self, mock_property):
        """Cannot read the state if the controller name has not been found."""
        self.portal.controller = None
        self.assertIsNone(self.portal.state)
        mock_property.assert_not_called()

    @mock.patch.object(nvmeof.Portal, 'get_device_by_property')
    def test_get_device(self, mock_property):
        """UUID has priority over everything else."""
        mock_property.return_value = 'result'
        self.target.nguid = 'nguid'
        res = self.portal.get_device()
        self.assertEqual('result', res)
        mock_property.assert_called_once_with('uuid', self.target.uuid)

    @mock.patch.object(nvmeof.Portal, 'get_device_by_property')
    def test_get_device_by_nguid(self, mock_property):
        """nguid takes priority over ns_id if no UUID."""
        mock_property.return_value = 'result'
        self.target.uuid = None
        self.target.nguid = 'nguid_value'
        self.target.ns_id = 'ns_id_value'
        res = self.portal.get_device()
        self.assertEqual('result', res)
        mock_property.assert_called_once_with('nguid', 'nguid_value')

    @mock.patch.object(nvmeof.Portal, 'get_device_by_property')
    def test_get_device_by_ns_id(self, mock_property):
        """ns_id takes priority if no UUID and nguid are present."""
        mock_property.return_value = 'result'
        self.target.uuid = None
        self.target.nguid = None
        self.target.ns_id = 'ns_id_value'
        res = self.portal.get_device()
        self.assertEqual('result', res)
        mock_property.assert_called_once_with('nsid', 'ns_id_value')

    @mock.patch.object(nvmeof.Target, 'get_device_path_by_initial_devices')
    @mock.patch.object(nvmeof.Portal, 'get_device_by_property')
    def test_get_device_by_initial_devices(self, mock_property, mock_get_dev):
        """With no id, calls target to get device from initial devices."""
        mock_get_dev.return_value = 'result'
        self.target.uuid = None
        self.target.nguid = None
        self.target.ns_id = None
        res = self.portal.get_device()
        self.assertEqual('result', res)
        mock_get_dev.assert_called_once_with()

    @mock.patch('glob.glob')
    def test_get_all_namespaces_ctrl_paths(self, mock_glob):
        expected = ['/sys/class/nvme-fabrics/ctl/nvme0/nvme0n1', '/sys/class/nvme-fabrics/ctl/nvme0/nvme1c1n2']
        mock_glob.return_value = expected[:]
        self.portal.controller = 'nvme0'
        res = self.portal.get_all_namespaces_ctrl_paths()
        self.assertEqual(expected, res)
        mock_glob.assert_called_once_with('/sys/class/nvme-fabrics/ctl/nvme0/nvme*')

    @mock.patch('glob.glob')
    def test_get_all_namespaces_ctrl_paths_no_controller(self, mock_glob):
        res = self.portal.get_all_namespaces_ctrl_paths()
        self.assertEqual([], res)
        mock_glob.assert_not_called()

    @mock.patch.object(nvmeof, 'nvme_basename', return_value='nvme1n2')
    @mock.patch.object(nvmeof, 'sysfs_property')
    @mock.patch.object(nvmeof.Portal, 'get_all_namespaces_ctrl_paths')
    def test_get_device_by_property(self, mock_paths, mock_property, mock_name):
        """Searches all devices for the right one and breaks when found."""
        mock_paths.return_value = ['/sys/class/nvme-fabrics/ctl/nvme0/nvme0n1', '/sys/class/nvme-fabrics/ctl/nvme0/nvme1c1n2', '/sys/class/nvme-fabrics/ctl/nvme0/nvme0n3']
        mock_property.side_effect = ['uuid1', 'uuid2']
        self.portal.controller = 'nvme0'
        res = self.portal.get_device_by_property('uuid', 'uuid2')
        self.assertEqual('/dev/nvme1n2', res)
        mock_paths.assert_called_once_with()
        self.assertEqual(2, mock_property.call_count)
        mock_property.assert_has_calls([mock.call('uuid', '/sys/class/nvme-fabrics/ctl/nvme0/nvme0n1'), mock.call('uuid', '/sys/class/nvme-fabrics/ctl/nvme0/nvme1c1n2')])
        mock_name.assert_called_once_with('/sys/class/nvme-fabrics/ctl/nvme0/nvme1c1n2')

    @mock.patch.object(nvmeof, 'nvme_basename', return_value='nvme1n2')
    @mock.patch.object(nvmeof, 'sysfs_property')
    @mock.patch.object(nvmeof.Portal, 'get_all_namespaces_ctrl_paths')
    def test_get_device_by_property_not_found(self, mock_paths, mock_property, mock_name):
        """Exhausts devices searching before returning None."""
        mock_paths.return_value = ['/sys/class/nvme-fabrics/ctl/nvme0/nvme0n1', '/sys/class/nvme-fabrics/ctl/nvme0/nvme0n2']
        mock_property.side_effect = ['uuid1', 'uuid2']
        self.portal.controller = 'nvme0'
        res = self.portal.get_device_by_property('uuid', 'uuid3')
        self.assertIsNone(res)
        mock_paths.assert_called_once_with()
        self.assertEqual(2, mock_property.call_count)
        mock_property.assert_has_calls([mock.call('uuid', '/sys/class/nvme-fabrics/ctl/nvme0/nvme0n1'), mock.call('uuid', '/sys/class/nvme-fabrics/ctl/nvme0/nvme0n2')])
        mock_name.assert_not_called()

    @mock.patch.object(nvmeof.Portal, 'get_all_namespaces_ctrl_paths')
    def test__can_disconnect_no_controller_name(self, mock_paths):
        """Cannot disconnect when portal doesn't have a controller."""
        res = self.portal.can_disconnect()
        self.assertFalse(res)
        mock_paths.assert_not_called()

    @ddt.data(([], True), (['/sys/class/nvme-fabrics/ctl/nvme0/nvme0n1', '/sys/class/nvme-fabrics/ctl/nvme0/nvme0n2'], False))
    @ddt.unpack
    @mock.patch.object(nvmeof.Portal, 'get_all_namespaces_ctrl_paths')
    def test__can_disconnect_not_1_namespace(self, ctrl_paths, expected, mock_paths):
        """Check if can disconnect when we don't have 1 namespace in subsys."""
        self.portal.controller = 'nvme0'
        mock_paths.return_value = ctrl_paths
        res = self.portal.can_disconnect()
        self.assertIs(expected, res)
        mock_paths.assert_called_once_with()

    @mock.patch.object(nvmeof.Portal, 'get_device')
    @mock.patch.object(nvmeof.Portal, 'get_all_namespaces_ctrl_paths')
    def test__can_disconnect(self, mock_paths, mock_device):
        """Can disconnect if the namespace is the one from this target.

        This tests that even when ANA is enabled it can identify the control
        path as belonging to the used device path.
        """
        self.portal.controller = 'nvme0'
        mock_device.return_value = '/dev/nvme1n2'
        mock_paths.return_value = ['/sys/class/nvme-fabrics/ctl/nvme0/nvme1c1n2']
        self.assertTrue(self.portal.can_disconnect())

    @mock.patch.object(nvmeof.Portal, 'get_device')
    @mock.patch.object(nvmeof.Portal, 'get_all_namespaces_ctrl_paths')
    def test__can_disconnect_different_target(self, mock_paths, mock_device):
        """Cannot disconnect if the namespace is from a different target."""
        self.portal.controller = 'nvme0'
        mock_device.return_value = None
        mock_paths.return_value = ['/sys/class/nvme-fabrics/ctl/nvme0/nvme1c1n2']
        self.assertFalse(self.portal.can_disconnect())