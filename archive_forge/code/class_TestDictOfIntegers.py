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
class TestDictOfIntegers(TestField):

    def setUp(self):
        super(TestDictOfIntegers, self).setUp()
        self.field = fields.DictOfIntegersField()
        self.coerce_good_values = [({'foo': '42'}, {'foo': 42}), ({'foo': 4.2}, {'foo': 4})]
        self.coerce_bad_values = [{1: 'bar'}, {'foo': 'boo'}, 'foo', {'foo': None}]
        self.to_primitive_values = [({'foo': 42}, {'foo': 42})]
        self.from_primitive_values = [({'foo': 42}, {'foo': 42})]

    def test_stringify(self):
        self.assertEqual('{key=42}', self.field.stringify({'key': 42}))