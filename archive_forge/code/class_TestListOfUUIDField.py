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
class TestListOfUUIDField(TestField):

    def setUp(self):
        super(TestListOfUUIDField, self).setUp()
        self.field = fields.ListOfUUIDField()
        self.uuid1 = '6b2097ea-d0e3-44dd-b131-95472b3ea8fd'
        self.uuid2 = '478c193d-2533-4e71-ab2b-c7683f67d7f9'
        self.coerce_good_values = [([self.uuid1, self.uuid2], [self.uuid1, self.uuid2])]
        self.to_primitive_values = [([self.uuid1], [self.uuid1])]
        self.from_primitive_values = [([self.uuid1], [self.uuid1])]

    def test_stringify(self):
        self.assertEqual('[%s,%s]' % (self.uuid1, self.uuid2), self.field.stringify([self.uuid1, self.uuid2]))