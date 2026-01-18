import re
from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils10
def _check_get_pci_device_address_None(self, return_code=0):
    pnp_device = mock.MagicMock()
    pnp_device.GetDeviceProperties.return_value = (return_code, [mock.MagicMock()])
    self._hostutils._conn_cimv2.Win32_PnPEntity.return_value = [pnp_device]
    pci_dev_address = self._hostutils._get_pci_device_address(mock.sentinel.pci_device_path)
    self.assertIsNone(pci_dev_address)