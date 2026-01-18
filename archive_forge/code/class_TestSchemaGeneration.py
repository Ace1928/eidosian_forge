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
class TestSchemaGeneration(test.TestCase):

    @base.VersionedObjectRegistry.register
    class FakeObject(base.VersionedObject):
        fields = {'a_boolean': fields.BooleanField(nullable=True)}

    @base.VersionedObjectRegistry.register
    class FakeComplexObject(base.VersionedObject):
        fields = {'a_dict': fields.DictOfListOfStringsField(), 'an_obj': fields.ObjectField('FakeObject', nullable=True), 'list_of_objs': fields.ListOfObjectsField('FakeObject')}

    def test_to_json_schema(self):
        schema = self.FakeObject.to_json_schema()
        self.assertEqual({'$schema': 'http://json-schema.org/draft-04/schema#', 'title': 'FakeObject', 'type': ['object'], 'properties': {'versioned_object.namespace': {'type': 'string'}, 'versioned_object.name': {'type': 'string'}, 'versioned_object.version': {'type': 'string'}, 'versioned_object.changes': {'type': 'array', 'items': {'type': 'string'}}, 'versioned_object.data': {'type': 'object', 'description': 'fields of FakeObject', 'properties': {'a_boolean': {'readonly': False, 'type': ['boolean', 'null']}}}}, 'required': ['versioned_object.namespace', 'versioned_object.name', 'versioned_object.version', 'versioned_object.data']}, schema)
        jsonschema.validate(self.FakeObject(a_boolean=True).obj_to_primitive(), self.FakeObject.to_json_schema())

    def test_to_json_schema_complex_object(self):
        schema = self.FakeComplexObject.to_json_schema()
        expected_schema = {'$schema': 'http://json-schema.org/draft-04/schema#', 'properties': {'versioned_object.changes': {'items': {'type': 'string'}, 'type': 'array'}, 'versioned_object.data': {'description': 'fields of FakeComplexObject', 'properties': {'a_dict': {'readonly': False, 'type': ['object'], 'additionalProperties': {'type': ['array'], 'readonly': False, 'items': {'type': ['string'], 'readonly': False}}}, 'an_obj': {'properties': {'versioned_object.changes': {'items': {'type': 'string'}, 'type': 'array'}, 'versioned_object.data': {'description': 'fields of FakeObject', 'properties': {'a_boolean': {'readonly': False, 'type': ['boolean', 'null']}}, 'type': 'object'}, 'versioned_object.name': {'type': 'string'}, 'versioned_object.namespace': {'type': 'string'}, 'versioned_object.version': {'type': 'string'}}, 'readonly': False, 'required': ['versioned_object.namespace', 'versioned_object.name', 'versioned_object.version', 'versioned_object.data'], 'type': ['object', 'null']}, 'list_of_objs': {'items': {'properties': {'versioned_object.changes': {'items': {'type': 'string'}, 'type': 'array'}, 'versioned_object.data': {'description': 'fields of FakeObject', 'properties': {'a_boolean': {'readonly': False, 'type': ['boolean', 'null']}}, 'type': 'object'}, 'versioned_object.name': {'type': 'string'}, 'versioned_object.namespace': {'type': 'string'}, 'versioned_object.version': {'type': 'string'}}, 'readonly': False, 'required': ['versioned_object.namespace', 'versioned_object.name', 'versioned_object.version', 'versioned_object.data'], 'type': ['object']}, 'readonly': False, 'type': ['array']}}, 'required': ['a_dict', 'list_of_objs'], 'type': 'object'}, 'versioned_object.name': {'type': 'string'}, 'versioned_object.namespace': {'type': 'string'}, 'versioned_object.version': {'type': 'string'}}, 'required': ['versioned_object.namespace', 'versioned_object.name', 'versioned_object.version', 'versioned_object.data'], 'title': 'FakeComplexObject', 'type': ['object']}
        self.assertEqual(expected_schema, schema)
        fake_obj = self.FakeComplexObject(a_dict={'key1': ['foo', 'bar'], 'key2': ['bar', 'baz']}, an_obj=self.FakeObject(a_boolean=True), list_of_objs=[self.FakeObject(a_boolean=False), self.FakeObject(a_boolean=True), self.FakeObject(a_boolean=False)])
        primitives = fake_obj.obj_to_primitive()
        jsonschema.validate(primitives, schema)