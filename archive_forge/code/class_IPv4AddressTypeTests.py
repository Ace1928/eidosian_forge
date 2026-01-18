import re
import unittest
from oslo_config import types
class IPv4AddressTypeTests(IPAddressTypeTests):
    type = types.IPAddress(4)

    def test_ipv6_address(self):
        self.assertInvalid('abcd:ef::1')