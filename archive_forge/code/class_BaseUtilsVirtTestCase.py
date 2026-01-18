from unittest import mock
import six
import importlib
from os_win.tests.unit import test_base
from os_win.utils import baseutils
class BaseUtilsVirtTestCase(test_base.OsWinBaseTestCase):
    """Unit tests for the os-win BaseUtilsVirt class."""

    def setUp(self):
        super(BaseUtilsVirtTestCase, self).setUp()
        self.utils = baseutils.BaseUtilsVirt()
        self.utils._conn_attr = mock.MagicMock()
        baseutils.BaseUtilsVirt._os_version = None
        mock.patch.object(importlib, 'util').start()

    @mock.patch.object(baseutils.BaseUtilsVirt, '_get_wmi_conn')
    def test_conn(self, mock_get_wmi_conn):
        self.utils._conn_attr = None
        self.assertEqual(mock_get_wmi_conn.return_value, self.utils._conn)
        mock_get_wmi_conn.assert_called_once_with(self.utils._wmi_namespace % '.')

    def test_vs_man_svc(self):
        mock_os = mock.MagicMock(Version='6.3.0')
        self._mock_wmi.WMI.return_value.Win32_OperatingSystem.return_value = [mock_os]
        expected = self.utils._conn.Msvm_VirtualSystemManagementService()[0]
        self.assertEqual(expected, self.utils._vs_man_svc)
        self.assertEqual(expected, self.utils._vs_man_svc_attr)

    @mock.patch.object(baseutils, 'wmi', create=True)
    def test_vs_man_svc_2012(self, mock_wmi):
        baseutils.BaseUtilsVirt._old_wmi = None
        mock_os = mock.MagicMock(Version='6.2.0')
        mock_wmi.WMI.return_value.Win32_OperatingSystem.return_value = [mock_os]
        fake_module_path = '/fake/path/to/module'
        mock_wmi.__path__ = [fake_module_path]
        spec = importlib.util.spec_from_file_location.return_value
        module = importlib.util.module_from_spec.return_value
        old_conn = module.WMI.return_value
        expected = old_conn.Msvm_VirtualSystemManagementService()[0]
        self.assertEqual(expected, self.utils._vs_man_svc)
        self.assertIsNone(self.utils._vs_man_svc_attr)
        importlib.util.spec_from_file_location.assert_called_once_with('old_wmi', '%s.py' % fake_module_path)
        spec.loader.exec_module.assert_called_once_with(module)
        importlib.util.module_from_spec.assert_called_once_with(importlib.util.spec_from_file_location.return_value)

    @mock.patch.object(baseutils.BaseUtilsVirt, '_get_wmi_compat_conn')
    def test_get_wmi_obj_compatibility_6_3(self, mock_get_wmi_compat):
        mock_os = mock.MagicMock(Version='6.3.0')
        self._mock_wmi.WMI.return_value.Win32_OperatingSystem.return_value = [mock_os]
        result = self.utils._get_wmi_obj(mock.sentinel.moniker, True)
        self.assertEqual(self._mock_wmi.WMI.return_value, result)

    @mock.patch.object(baseutils.BaseUtilsVirt, '_get_wmi_compat_conn')
    def test_get_wmi_obj_no_compatibility_6_2(self, mock_get_wmi_compat):
        baseutils.BaseUtilsVirt._os_version = [6, 2]
        result = self.utils._get_wmi_obj(mock.sentinel.moniker, False)
        self.assertEqual(self._mock_wmi.WMI.return_value, result)

    @mock.patch.object(baseutils.BaseUtilsVirt, '_get_wmi_compat_conn')
    def test_get_wmi_obj_compatibility_6_2(self, mock_get_wmi_compat):
        baseutils.BaseUtilsVirt._os_version = [6, 2]
        result = self.utils._get_wmi_obj(mock.sentinel.moniker, True)
        self.assertEqual(mock_get_wmi_compat.return_value, result)