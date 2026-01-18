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
class TestVersionPredicate(TestString):

    def setUp(self):
        super(TestVersionPredicate, self).setUp()
        self.field = fields.VersionPredicateField()
        self.coerce_good_values = [('>=1.0', '>=1.0'), ('==1.1', '==1.1'), ('<1.1.0', '<1.1.0')]
        self.coerce_bad_values = ['1', 'foo', '>1', 1.0, '1.0', '=1.0']
        self.to_primitive_values = self.coerce_good_values[0:1]
        self.from_primitive_values = self.coerce_good_values[0:1]