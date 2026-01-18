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
class TestIPV4Network(TestField):

    def setUp(self):
        super(TestIPV4Network, self).setUp()
        self.field = fields.IPV4NetworkField()
        self.coerce_good_values = [('1.2.3.4/24', netaddr.IPNetwork('1.2.3.4/24'))]
        self.coerce_bad_values = ['foo', '::1/32']
        self.to_primitive_values = [(netaddr.IPNetwork('1.2.3.4/24'), '1.2.3.4/24')]
        self.from_primitive_values = [('1.2.3.4/24', netaddr.IPNetwork('1.2.3.4/24'))]

    def test_get_schema(self):
        schema = self.field.get_schema()
        self.assertEqual(['string'], schema['type'])
        self.assertEqual(False, schema['readonly'])
        pattern = schema['pattern']
        for _, valid_val in self.coerce_good_values:
            self.assertRegex(str(valid_val), pattern)
        invalid_vals = [x for x in self.coerce_bad_values]
        for invalid_val in invalid_vals:
            self.assertNotRegex(str(invalid_val), pattern)