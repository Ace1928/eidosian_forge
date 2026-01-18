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
class TestListOfIntegers(TestField):

    def setUp(self):
        super(TestListOfIntegers, self).setUp()
        self.field = fields.ListOfIntegersField()
        self.coerce_good_values = [(['1', 2], [1, 2]), ([1, 2], [1, 2])]
        self.coerce_bad_values = [['foo']]
        self.to_primitive_values = [([1], [1])]
        self.from_primitive_values = [([1], [1])]

    def test_stringify(self):
        self.assertEqual('[[1, 2]]', self.field.stringify([[1, 2]]))