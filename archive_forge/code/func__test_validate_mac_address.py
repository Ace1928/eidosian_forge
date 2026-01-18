import string
from unittest import mock
import netaddr
from neutron_lib._i18n import _
from neutron_lib.api import converters
from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib import fixture
from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def _test_validate_mac_address(self, validator, allow_none=False):
    mac_addr = 'ff:16:3e:4f:00:00'
    msg = validator(mac_addr)
    self.assertIsNone(msg)
    mac_addr = 'ffa:16:3e:4f:00:00'
    msg = validator(mac_addr)
    err_msg = "'%s' is not a valid MAC address"
    self.assertEqual(err_msg % mac_addr, msg)
    for invalid_mac_addr in constants.INVALID_MAC_ADDRESSES:
        msg = validator(invalid_mac_addr)
        self.assertEqual(err_msg % invalid_mac_addr, msg)
    mac_addr = '123'
    msg = validator(mac_addr)
    self.assertEqual(err_msg % mac_addr, msg)
    mac_addr = None
    msg = validator(mac_addr)
    if allow_none:
        self.assertIsNone(msg)
    else:
        self.assertEqual(err_msg % mac_addr, msg)
    mac_addr = 'ff:16:3e:4f:00:00\r'
    msg = validator(mac_addr)
    self.assertEqual(err_msg % mac_addr, msg)