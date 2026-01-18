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
class TestSetOfIntegers(TestField):

    def setUp(self):
        super(TestSetOfIntegers, self).setUp()
        self.field = fields.SetOfIntegersField()
        self.coerce_good_values = [(set(['1', 2]), set([1, 2]))]
        self.coerce_bad_values = [set(['foo'])]
        self.to_primitive_values = [(set([1]), tuple([1]))]
        self.from_primitive_values = [(tuple([1]), set([1]))]

    def test_stringify(self):
        self.assertEqual('set([1,2])', self.field.stringify(set([1, 2])))

    def test_repr(self):
        self.assertEqual("Set(default=<class 'oslo_versionedobjects.fields.UnspecifiedDefault'>,nullable=False)", repr(self.field))
        self.assertEqual('Set(default=set([]),nullable=False)', repr(fields.SetOfIntegersField(default=set())))
        self.assertEqual('Set(default=set([1,a]),nullable=False)', repr(fields.SetOfIntegersField(default={1, 'a'})))