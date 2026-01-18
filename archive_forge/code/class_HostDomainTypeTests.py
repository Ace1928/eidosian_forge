import re
import unittest
from oslo_config import types
class HostDomainTypeTests(TypeTestHelper, unittest.TestCase):
    type = types.HostDomain()

    def test_invalid_host_addresses(self):
        self.assertInvalid('-1')
        self.assertInvalid('3.14')
        self.assertInvalid('10.0')
        self.assertInvalid('host..name')
        self.assertInvalid('org.10')
        self.assertInvalid('0.0.00')

    def test_valid_host_addresses(self):
        self.assertConvertedValue('_foo', '_foo')
        self.assertConvertedValue('host_name', 'host_name')
        self.assertConvertedValue('overcloud-novacompute_edge1-0.internalapi.localdomain', 'overcloud-novacompute_edge1-0.internalapi.localdomain')
        self.assertConvertedValue('host_01.co.uk', 'host_01.co.uk')
        self.assertConvertedValue('_site01001', '_site01001')