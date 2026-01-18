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
class TestPCIAddress(TestField):

    def setUp(self):
        super(TestPCIAddress, self).setUp()
        self.field = fields.PCIAddressField()
        self.coerce_good_values = [('0000:02:00.0', '0000:02:00.0'), ('FFFF:FF:1F.7', 'ffff:ff:1f.7'), ('fFfF:fF:1F.7', 'ffff:ff:1f.7')]
        self.coerce_bad_values = ['000:02:00.0', '00000:02:00.0', 'FFFF:FF:2F.7', 'FFFF:GF:1F.7', 1123123, {}]
        self.to_primitive_values = self.coerce_good_values[0:1]
        self.from_primitive_values = self.coerce_good_values[0:1]

    def test_get_schema(self):
        schema = self.field.get_schema()
        self.assertEqual(['string'], schema['type'])
        self.assertEqual(False, schema['readonly'])
        pattern = schema['pattern']
        for _, valid_val in self.coerce_good_values:
            self.assertRegex(valid_val, pattern)
        invalid_vals = [x for x in self.coerce_bad_values if isinstance(x, str)]
        for invalid_val in invalid_vals:
            self.assertNotRegex(invalid_val, pattern)