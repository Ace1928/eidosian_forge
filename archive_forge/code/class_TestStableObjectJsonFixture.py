import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
class TestStableObjectJsonFixture(test.TestCase):

    def test_changes_sort(self):

        @base.VersionedObjectRegistry.register_if(False)
        class TestObject(base.VersionedObject):
            fields = {'z': fields.StringField(), 'a': fields.StringField()}

            def obj_what_changed(self):
                return ['z', 'a']
        obj = TestObject(a='foo', z='bar')
        self.assertEqual(['z', 'a'], obj.obj_to_primitive()['versioned_object.changes'])
        with fixture.StableObjectJsonFixture():
            self.assertEqual(['a', 'z'], obj.obj_to_primitive()['versioned_object.changes'])