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
class TestIPNetwork(TestField):

    def setUp(self):
        super(TestIPNetwork, self).setUp()
        self.field = fields.IPNetworkField()
        self.coerce_good_values = [('::1/0', netaddr.IPNetwork('::1/0')), ('1.2.3.4/24', netaddr.IPNetwork('1.2.3.4/24')), (netaddr.IPNetwork('::1/32'), netaddr.IPNetwork('::1/32'))]
        self.coerce_bad_values = ['foo']
        self.to_primitive_values = [(netaddr.IPNetwork('::1/0'), '::1/0')]
        self.from_primitive_values = [('::1/0', netaddr.IPNetwork('::1/0'))]