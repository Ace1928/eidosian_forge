from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
class TestConvertIPv6AddrCanonicalFormat(base.BaseTestCase):

    def test_convert_ipv6_address_extended_add_with_zeroes(self):
        result = converters.convert_ip_to_canonical_format('2001:0db8:0:0:0:0:0:0001')
        self.assertEqual('2001:db8::1', result)

    @testtools.skipIf(tools.is_bsd(), 'bug/1484837')
    def test_convert_ipv6_compressed_address_OSX_skip(self):
        result = converters.convert_ip_to_canonical_format('2001:db8:0:1:1:1:1:1')
        self.assertEqual('2001:db8:0:1:1:1:1:1', result)

    def test_convert_ipv6_extended_addr_to_compressed(self):
        result = converters.convert_ip_to_canonical_format(u'Fe80:0:0:0:0:0:0:1')
        self.assertEqual('fe80::1', result)

    def test_convert_ipv4_address(self):
        result = converters.convert_ip_to_canonical_format(u'192.168.1.1')
        self.assertEqual('192.168.1.1', result)

    def test_convert_None_address(self):
        result = converters.convert_ip_to_canonical_format(None)
        self.assertIsNone(result)

    def test_convert_invalid_address(self):
        result = converters.convert_ip_to_canonical_format('on')
        self.assertEqual('on', result)
        result = converters.convert_ip_to_canonical_format('192.168.1.1/32')
        self.assertEqual('192.168.1.1/32', result)
        result = converters.convert_ip_to_canonical_format('2001:db8:0:1:1:1:1:1/128')
        self.assertEqual('2001:db8:0:1:1:1:1:1/128', result)