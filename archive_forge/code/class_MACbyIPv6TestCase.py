import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
class MACbyIPv6TestCase(test_base.BaseTestCase):
    """Unit tests to extract MAC from IPv6."""

    def test_reverse_generate_IPv6_by_EUI64(self):
        self.assertEqual(netaddr.EUI('00:16:3e:33:44:55'), netutils.get_mac_addr_by_ipv6(netaddr.IPAddress('2001:db8::216:3eff:fe33:4455')))

    def test_random_qemu_mac(self):
        self.assertEqual(netaddr.EUI('52:54:00:42:02:19'), netutils.get_mac_addr_by_ipv6(netaddr.IPAddress('fe80::5054:ff:fe42:219')))

    def test_local(self):
        self.assertEqual(netaddr.EUI('02:00:00:00:00:00'), netutils.get_mac_addr_by_ipv6(netaddr.IPAddress('fe80::ff:fe00:0')))

    def test_universal(self):
        self.assertEqual(netaddr.EUI('00:00:00:00:00:00'), netutils.get_mac_addr_by_ipv6(netaddr.IPAddress('fe80::200:ff:fe00:0')))