import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
class TestTimestampedObject(test.TestCase):
    """Test TimestampedObject mixin.

    Do this by creating an object that uses the mixin and confirm that the
    added fields are there and in fact behaves as the DateTimeFields we desire.
    """

    def setUp(self):
        super(TestTimestampedObject, self).setUp()

        @base.VersionedObjectRegistry.register_if(False)
        class MyTimestampedObject(base.VersionedObject, base.TimestampedObject):
            fields = {'field1': fields.Field(fields.String())}
        self.myclass = MyTimestampedObject
        self.my_object = self.myclass(field1='field1')

    def test_timestamped_has_fields(self):
        self.assertEqual('field1', self.my_object.field1)
        self.assertIn('updated_at', self.my_object.fields)
        self.assertIn('created_at', self.my_object.fields)

    def test_timestamped_holds_timestamps(self):
        now = timeutils.utcnow(with_timezone=True)
        self.my_object.updated_at = now
        self.my_object.created_at = now
        self.assertEqual(now, self.my_object.updated_at)
        self.assertEqual(now, self.my_object.created_at)

    def test_timestamped_rejects_not_timestamps(self):
        with testtools.ExpectedException(ValueError, '.*parse date.*'):
            self.my_object.updated_at = 'a string'
        with testtools.ExpectedException(ValueError, '.*parse date.*'):
            self.my_object.created_at = 'a string'