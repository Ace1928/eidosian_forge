from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
class TestConvertToSanitizedMacAddress(base.BaseTestCase):

    def test_sanitize_mac_address(self):
        input_exp = (('00:11:22:33:44:55', '00:11:22:33:44:55'), ('00:11:22:33:44:5', '00:11:22:33:44:05'), ('0:1:2:3:4:5', '00:01:02:03:04:05'), ('ca:FE:cA:Fe:a:E', 'ca:fe:ca:fe:0a:0e'), ('12345678901', '01:23:45:67:89:01'), ('012345678901', '01:23:45:67:89:01'))
        for input, expected in input_exp:
            self.assertEqual(expected, converters.convert_to_sanitized_mac_address(input))
            eui_address = netaddr.EUI(input)
            self.assertEqual(expected, converters.convert_to_sanitized_mac_address(eui_address))
        self.assertEqual('00:11:22:33:44', converters.convert_to_sanitized_mac_address('00:11:22:33:44'))
        self.assertEqual('00:11:22:33:44:', converters.convert_to_sanitized_mac_address('00:11:22:33:44:'))