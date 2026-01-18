from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
class TestConvertProtocol(base.BaseTestCase):

    def test_tcp_is_valid(self):
        result = converters.convert_to_protocol(constants.PROTO_NAME_TCP)
        self.assertEqual(constants.PROTO_NAME_TCP, result)
        proto_num_str = str(constants.PROTO_NUM_TCP)
        result = converters.convert_to_protocol(proto_num_str)
        self.assertEqual(proto_num_str, result)

    def test_udp_is_valid(self):
        result = converters.convert_to_protocol(constants.PROTO_NAME_UDP)
        self.assertEqual(constants.PROTO_NAME_UDP, result)
        proto_num_str = str(constants.PROTO_NUM_UDP)
        result = converters.convert_to_protocol(proto_num_str)
        self.assertEqual(proto_num_str, result)

    def test_icmp_is_valid(self):
        result = converters.convert_to_protocol(constants.PROTO_NAME_ICMP)
        self.assertEqual(constants.PROTO_NAME_ICMP, result)
        proto_num_str = str(constants.PROTO_NUM_ICMP)
        result = converters.convert_to_protocol(proto_num_str)
        self.assertEqual(proto_num_str, result)

    def test_numeric_is_valid(self):
        proto_num_str = str(constants.PROTO_NUM_IGMP)
        result = converters.convert_to_protocol(proto_num_str)
        self.assertEqual(proto_num_str, result)

    def test_numeric_too_high(self):
        with testtools.ExpectedException(n_exc.InvalidInput):
            converters.convert_to_protocol('300')

    def test_numeric_too_low(self):
        with testtools.ExpectedException(n_exc.InvalidInput):
            converters.convert_to_protocol('-1')

    def test_unknown_string(self):
        with testtools.ExpectedException(n_exc.InvalidInput):
            converters.convert_to_protocol('Invalid')