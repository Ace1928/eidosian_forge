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
class TestDictOfStringsNone(TestField):

    def setUp(self):
        super(TestDictOfStringsNone, self).setUp()
        self.field = fields.DictOfNullableStringsField()
        self.coerce_good_values = [({'foo': 'bar'}, {'foo': 'bar'}), ({'foo': 1}, {'foo': '1'}), ({'foo': None}, {'foo': None})]
        self.coerce_bad_values = [{1: 'bar'}, 'foo']
        self.to_primitive_values = [({'foo': 'bar'}, {'foo': 'bar'})]
        self.from_primitive_values = [({'foo': 'bar'}, {'foo': 'bar'})]

    def test_stringify(self):
        self.assertEqual("{k2=None,key='val'}", self.field.stringify({'k2': None, 'key': 'val'}))