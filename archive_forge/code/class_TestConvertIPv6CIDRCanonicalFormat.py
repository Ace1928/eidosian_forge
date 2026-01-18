from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
class TestConvertIPv6CIDRCanonicalFormat(base.BaseTestCase):

    def test_convert_ipv4_address_with_CIDR(self):
        result = converters.convert_cidr_to_canonical_format('192.168.1.1/24')
        self.assertEqual('192.168.1.1/24', result)

    def test_convert_ipv6_extended_addr_withcidr_to_compressed(self):
        result = converters.convert_cidr_to_canonical_format('Fe80:0:0:0:0:0:0:1/64')
        self.assertEqual('fe80::1/64', result)

    def test_convert_non_ip_addr_with_slash(self):
        with testtools.ExpectedException(n_exc.InvalidInput):
            converters.convert_cidr_to_canonical_format('Dormamu/DarkSeid/Vulture')