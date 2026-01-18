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
class TestListOfSetsOfIntegers(TestField):

    def setUp(self):
        super(TestListOfSetsOfIntegers, self).setUp()
        self.field = fields.ListOfSetsOfIntegersField()
        self.coerce_good_values = [([set(['1', 2]), set([3, '4'])], [set([1, 2]), set([3, 4])])]
        self.coerce_bad_values = [[set(['foo'])]]
        self.to_primitive_values = [([set([1])], [tuple([1])])]
        self.from_primitive_values = [([tuple([1])], [set([1])])]

    def test_stringify(self):
        self.assertEqual('[set([1,2])]', self.field.stringify([set([1, 2])]))