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
class TestDictOfListOfStrings(TestField):

    def setUp(self):
        super(TestDictOfListOfStrings, self).setUp()
        self.field = fields.DictOfListOfStringsField()
        self.coerce_good_values = [({'foo': ['1', '2']}, {'foo': ['1', '2']}), ({'foo': [1]}, {'foo': ['1']})]
        self.coerce_bad_values = [{'foo': [None, None]}, 'foo']
        self.to_primitive_values = [({'foo': ['1', '2']}, {'foo': ['1', '2']})]
        self.from_primitive_values = [({'foo': ['1', '2']}, {'foo': ['1', '2']})]

    def test_stringify(self):
        self.assertEqual("{foo=['1','2']}", self.field.stringify({'foo': ['1', '2']}))