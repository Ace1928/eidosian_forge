import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
class TestIPV4AndV6Address(TestField):

    def setUp(self):
        super(TestIPV4AndV6Address, self).setUp()
        self.field = fields.IPV4AndV6Address()
        self.coerce_good_values = [('::1', netaddr.IPAddress('::1')), (netaddr.IPAddress('::1'), netaddr.IPAddress('::1')), ('1.2.3.4', netaddr.IPAddress('1.2.3.4')), (netaddr.IPAddress('1.2.3.4'), netaddr.IPAddress('1.2.3.4'))]
        self.coerce_bad_values = ['1-2', 'foo']
        self.to_primitive_values = [(netaddr.IPAddress('::1'), '::1'), (netaddr.IPAddress('1.2.3.4'), '1.2.3.4')]
        self.from_primitive_values = [('::1', netaddr.IPAddress('::1')), ('1.2.3.4', netaddr.IPAddress('1.2.3.4'))]

    def test_get_schema(self):
        self.assertEqual({'oneOf': [{'format': 'ipv4', 'type': ['string']}, {'format': 'ipv6', 'type': ['string']}]}, self.field.get_schema())