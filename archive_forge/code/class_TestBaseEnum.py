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
class TestBaseEnum(TestField):

    def setUp(self):
        super(TestBaseEnum, self).setUp()
        self.field = FakeEnumField()
        self.coerce_good_values = [('frog', 'frog'), ('platypus', 'platypus'), ('alligator', 'alligator')]
        self.coerce_bad_values = ['aardvark', 'wookie']
        self.to_primitive_values = self.coerce_good_values[0:1]
        self.from_primitive_values = self.coerce_good_values[0:1]

    def test_stringify(self):
        self.assertEqual("'platypus'", self.field.stringify('platypus'))

    def test_stringify_invalid(self):
        self.assertRaises(ValueError, self.field.stringify, 'aardvark')

    def test_fingerprint(self):
        field1 = FakeEnumField()
        field2 = FakeEnumAltField()
        self.assertNotEqual(str(field1), str(field2))

    def test_valid_values(self):
        self.assertEqual(self.field.valid_values, FakeEnum.ALL)

    def test_valid_values_keeps_type(self):
        self.assertIsInstance(self.field.valid_values, tuple)
        self.assertIsInstance(FakeEnumAltField().valid_values, set)