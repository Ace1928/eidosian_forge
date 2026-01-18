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
class ConnectorUtilsTestCase(test_base.TestCase):

    @mock.patch.object(nvmeof.NVMeOFConnector, '_is_native_multipath_supported', return_value=False)
    @mock.patch.object(priv_nvmeof, 'get_system_uuid', return_value=None)
    @mock.patch.object(nvmeof.NVMeOFConnector, '_get_host_uuid', return_value=None)
    @mock.patch.object(utils, 'get_host_nqn', return_value=None)
    @mock.patch.object(iscsi.ISCSIConnector, 'get_initiator', return_value='fakeinitiator')
    @mock.patch.object(linuxfc.LinuxFibreChannel, 'get_fc_wwpns', return_value=None)
    @mock.patch.object(linuxfc.LinuxFibreChannel, 'get_fc_wwnns', return_value=None)
    @mock.patch.object(platform, 'machine', mock.Mock(return_value='s390x'))
    @mock.patch('sys.platform', 'linux2')
    @mock.patch.object(utils, 'get_nvme_host_id', mock.Mock(return_value=None))
    def _test_brick_get_connector_properties(self, multipath, enforce_multipath, multipath_result, mock_wwnns, mock_wwpns, mock_initiator, mock_nqn, mock_hostuuid, mock_sysuuid, mock_native_multipath_supported, host='fakehost'):
        props_actual = connector.get_connector_properties('sudo', MY_IP, multipath, enforce_multipath, host=host)
        os_type = 'linux2'
        platform = 's390x'
        props = {'initiator': 'fakeinitiator', 'host': host, 'ip': MY_IP, 'multipath': multipath_result, 'nvme_native_multipath': False, 'os_type': os_type, 'platform': platform, 'do_local_attach': False}
        self.assertEqual(props, props_actual)

    def test_brick_get_connector_properties_connectors_called(self):
        """Make sure every connector is called."""
        mock_list = []
        for item in connector._get_connector_list():
            patched = mock.MagicMock()
            patched.platform = platform.machine()
            patched.os_type = sys.platform
            patched.__name__ = item
            patched.get_connector_properties.return_value = {}
            patcher = mock.patch(item, new=patched)
            patcher.start()
            self.addCleanup(patcher.stop)
            mock_list.append(patched)
        connector.get_connector_properties('sudo', MY_IP, True, True)
        for item in mock_list:
            assert item.get_connector_properties.called

    def test_brick_get_connector_properties(self):
        self._test_brick_get_connector_properties(False, False, False)

    @mock.patch.object(priv_rootwrap, 'custom_execute', side_effect=OSError(2))
    @mock.patch.object(priv_rootwrap, 'execute', return_value=('', ''))
    def test_brick_get_connector_properties_multipath(self, mock_execute, mock_custom_execute):
        self._test_brick_get_connector_properties(True, True, True)
        mock_execute.assert_called_once_with('multipathd', 'show', 'status', run_as_root=True, root_helper='sudo')
        mock_custom_execute.assert_called_once_with('nvme', 'version')

    @mock.patch.object(priv_rootwrap, 'custom_execute', side_effect=OSError(2))
    @mock.patch.object(priv_rootwrap, 'execute', side_effect=putils.ProcessExecutionError)
    def test_brick_get_connector_properties_fallback(self, mock_execute, mock_custom_execute):
        self._test_brick_get_connector_properties(True, False, False)
        mock_execute.assert_called_once_with('multipathd', 'show', 'status', run_as_root=True, root_helper='sudo')
        mock_custom_execute.assert_called_once_with('nvme', 'version')

    @mock.patch.object(priv_rootwrap, 'execute', side_effect=putils.ProcessExecutionError)
    def test_brick_get_connector_properties_raise(self, mock_execute):
        self.assertRaises(putils.ProcessExecutionError, self._test_brick_get_connector_properties, True, True, None)

    def test_brick_connector_properties_override_hostname(self):
        override_host = 'myhostname'
        self._test_brick_get_connector_properties(False, False, False, host=override_host)