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
class TestNamespaceCompatibility(test.TestCase):

    def setUp(self):
        super(TestNamespaceCompatibility, self).setUp()

        @base.VersionedObjectRegistry.register_if(False)
        class TestObject(base.VersionedObject):
            OBJ_SERIAL_NAMESPACE = 'foo'
            OBJ_PROJECT_NAMESPACE = 'tests'
        self.test_class = TestObject

    def test_obj_primitive_key(self):
        self.assertEqual('foo.data', self.test_class._obj_primitive_key('data'))

    def test_obj_primitive_field(self):
        primitive = {'foo.data': mock.sentinel.data}
        self.assertEqual(mock.sentinel.data, self.test_class._obj_primitive_field(primitive, 'data'))

    def test_obj_primitive_field_namespace(self):
        primitive = {'foo.name': 'TestObject', 'foo.namespace': 'tests', 'foo.version': '1.0', 'foo.data': {}}
        with mock.patch.object(self.test_class, 'obj_class_from_name'):
            self.test_class.obj_from_primitive(primitive)

    def test_obj_primitive_field_namespace_wrong(self):
        primitive = {'foo.name': 'TestObject', 'foo.namespace': 'wrong', 'foo.version': '1.0', 'foo.data': {}}
        self.assertRaises(exception.UnsupportedObjectError, self.test_class.obj_from_primitive, primitive)