from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
def _check_create_lookup_record(self, customer_addr, expected_type):
    lookup = mock.MagicMock()
    scimv2 = self.utils._scimv2
    obj_class = scimv2.MSFT_NetVirtualizationLookupRecordSettingData
    obj_class.return_value = [lookup]
    self.utils.create_lookup_record(mock.sentinel.provider_addr, customer_addr, mock.sentinel.mac_addr, mock.sentinel.fake_vsid)
    self.assertTrue(lookup.Delete_.called)
    obj_class.new.assert_called_once_with(VirtualSubnetID=mock.sentinel.fake_vsid, Rule=self.utils._TRANSLATE_ENCAP, Type=expected_type, MACAddress=mock.sentinel.mac_addr, CustomerAddress=customer_addr, ProviderAddress=mock.sentinel.provider_addr)