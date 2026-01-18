import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
class IPv6byEUI64TestCase(test_base.BaseTestCase):
    """Unit tests to generate IPv6 by EUI-64 operations."""

    def test_generate_IPv6_by_EUI64(self):
        addr = netutils.get_ipv6_addr_by_EUI64('2001:db8::', '00:16:3e:33:44:55')
        self.assertEqual('2001:db8::216:3eff:fe33:4455', addr.format())

    def test_generate_IPv6_with_IPv4_prefix(self):
        ipv4_prefix = '10.0.8'
        mac = '00:16:3e:33:44:55'
        self.assertRaises(ValueError, lambda: netutils.get_ipv6_addr_by_EUI64(ipv4_prefix, mac))

    def test_generate_IPv6_with_bad_mac(self):
        bad_mac = '00:16:3e:33:44:5Z'
        prefix = '2001:db8::'
        self.assertRaises(ValueError, lambda: netutils.get_ipv6_addr_by_EUI64(prefix, bad_mac))

    def test_generate_IPv6_with_bad_prefix(self):
        mac = '00:16:3e:33:44:55'
        bad_prefix = 'bb'
        self.assertRaises(ValueError, lambda: netutils.get_ipv6_addr_by_EUI64(bad_prefix, mac))

    def test_generate_IPv6_with_error_prefix_type(self):
        mac = '00:16:3e:33:44:55'
        prefix = 123
        self.assertRaises(TypeError, lambda: netutils.get_ipv6_addr_by_EUI64(prefix, mac))

    def test_generate_IPv6_with_empty_prefix(self):
        mac = '00:16:3e:33:44:55'
        prefix = ''
        self.assertRaises(ValueError, lambda: netutils.get_ipv6_addr_by_EUI64(prefix, mac))