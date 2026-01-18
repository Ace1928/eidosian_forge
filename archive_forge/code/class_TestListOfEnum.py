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
class TestListOfEnum(TestField):

    def setUp(self):
        super(TestListOfEnum, self).setUp()
        self.field = fields.ListOfEnumField(valid_values=['foo', 'bar'])
        self.coerce_good_values = [(['foo', 'bar'], ['foo', 'bar'])]
        self.coerce_bad_values = ['foo', ['foo', 'bar1']]
        self.to_primitive_values = [(['foo'], ['foo'])]
        self.from_primitive_values = [(['foo'], ['foo'])]

    def test_stringify(self):
        self.assertEqual("['foo']", self.field.stringify(['foo']))

    def test_stringify_invalid(self):
        self.assertRaises(ValueError, self.field.stringify, '[abc]')

    def test_fingerprint(self):
        field1 = fields.ListOfEnumField(valid_values=['foo', 'bar'])
        field2 = fields.ListOfEnumField(valid_values=['foo', 'bar1'])
        self.assertNotEqual(str(field1), str(field2))