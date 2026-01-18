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
class TestFieldType(test.TestCase):

    def test_get_schema(self):
        self.assertRaises(NotImplementedError, fields.FieldType().get_schema)