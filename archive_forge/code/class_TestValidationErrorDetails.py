from __future__ import annotations
from collections import deque, namedtuple
from contextlib import contextmanager
from decimal import Decimal
from io import BytesIO
from typing import Any
from unittest import TestCase, mock
from urllib.request import pathname2url
import json
import os
import sys
import tempfile
import warnings
from attrs import define, field
from referencing.jsonschema import DRAFT202012
import referencing.exceptions
from jsonschema import (
class TestValidationErrorDetails(TestCase):

    def test_anyOf(self):
        instance = 5
        schema = {'anyOf': [{'minimum': 20}, {'type': 'string'}]}
        validator = validators.Draft4Validator(schema)
        errors = list(validator.iter_errors(instance))
        self.assertEqual(len(errors), 1)
        e = errors[0]
        self.assertEqual(e.validator, 'anyOf')
        self.assertEqual(e.validator_value, schema['anyOf'])
        self.assertEqual(e.instance, instance)
        self.assertEqual(e.schema, schema)
        self.assertIsNone(e.parent)
        self.assertEqual(e.path, deque([]))
        self.assertEqual(e.relative_path, deque([]))
        self.assertEqual(e.absolute_path, deque([]))
        self.assertEqual(e.json_path, '$')
        self.assertEqual(e.schema_path, deque(['anyOf']))
        self.assertEqual(e.relative_schema_path, deque(['anyOf']))
        self.assertEqual(e.absolute_schema_path, deque(['anyOf']))
        self.assertEqual(len(e.context), 2)
        e1, e2 = sorted_errors(e.context)
        self.assertEqual(e1.validator, 'minimum')
        self.assertEqual(e1.validator_value, schema['anyOf'][0]['minimum'])
        self.assertEqual(e1.instance, instance)
        self.assertEqual(e1.schema, schema['anyOf'][0])
        self.assertIs(e1.parent, e)
        self.assertEqual(e1.path, deque([]))
        self.assertEqual(e1.absolute_path, deque([]))
        self.assertEqual(e1.relative_path, deque([]))
        self.assertEqual(e1.json_path, '$')
        self.assertEqual(e1.schema_path, deque([0, 'minimum']))
        self.assertEqual(e1.relative_schema_path, deque([0, 'minimum']))
        self.assertEqual(e1.absolute_schema_path, deque(['anyOf', 0, 'minimum']))
        self.assertFalse(e1.context)
        self.assertEqual(e2.validator, 'type')
        self.assertEqual(e2.validator_value, schema['anyOf'][1]['type'])
        self.assertEqual(e2.instance, instance)
        self.assertEqual(e2.schema, schema['anyOf'][1])
        self.assertIs(e2.parent, e)
        self.assertEqual(e2.path, deque([]))
        self.assertEqual(e2.relative_path, deque([]))
        self.assertEqual(e2.absolute_path, deque([]))
        self.assertEqual(e2.json_path, '$')
        self.assertEqual(e2.schema_path, deque([1, 'type']))
        self.assertEqual(e2.relative_schema_path, deque([1, 'type']))
        self.assertEqual(e2.absolute_schema_path, deque(['anyOf', 1, 'type']))
        self.assertEqual(len(e2.context), 0)

    def test_type(self):
        instance = {'foo': 1}
        schema = {'type': [{'type': 'integer'}, {'type': 'object', 'properties': {'foo': {'enum': [2]}}}]}
        validator = validators.Draft3Validator(schema)
        errors = list(validator.iter_errors(instance))
        self.assertEqual(len(errors), 1)
        e = errors[0]
        self.assertEqual(e.validator, 'type')
        self.assertEqual(e.validator_value, schema['type'])
        self.assertEqual(e.instance, instance)
        self.assertEqual(e.schema, schema)
        self.assertIsNone(e.parent)
        self.assertEqual(e.path, deque([]))
        self.assertEqual(e.relative_path, deque([]))
        self.assertEqual(e.absolute_path, deque([]))
        self.assertEqual(e.json_path, '$')
        self.assertEqual(e.schema_path, deque(['type']))
        self.assertEqual(e.relative_schema_path, deque(['type']))
        self.assertEqual(e.absolute_schema_path, deque(['type']))
        self.assertEqual(len(e.context), 2)
        e1, e2 = sorted_errors(e.context)
        self.assertEqual(e1.validator, 'type')
        self.assertEqual(e1.validator_value, schema['type'][0]['type'])
        self.assertEqual(e1.instance, instance)
        self.assertEqual(e1.schema, schema['type'][0])
        self.assertIs(e1.parent, e)
        self.assertEqual(e1.path, deque([]))
        self.assertEqual(e1.relative_path, deque([]))
        self.assertEqual(e1.absolute_path, deque([]))
        self.assertEqual(e1.json_path, '$')
        self.assertEqual(e1.schema_path, deque([0, 'type']))
        self.assertEqual(e1.relative_schema_path, deque([0, 'type']))
        self.assertEqual(e1.absolute_schema_path, deque(['type', 0, 'type']))
        self.assertFalse(e1.context)
        self.assertEqual(e2.validator, 'enum')
        self.assertEqual(e2.validator_value, [2])
        self.assertEqual(e2.instance, 1)
        self.assertEqual(e2.schema, {'enum': [2]})
        self.assertIs(e2.parent, e)
        self.assertEqual(e2.path, deque(['foo']))
        self.assertEqual(e2.relative_path, deque(['foo']))
        self.assertEqual(e2.absolute_path, deque(['foo']))
        self.assertEqual(e2.json_path, '$.foo')
        self.assertEqual(e2.schema_path, deque([1, 'properties', 'foo', 'enum']))
        self.assertEqual(e2.relative_schema_path, deque([1, 'properties', 'foo', 'enum']))
        self.assertEqual(e2.absolute_schema_path, deque(['type', 1, 'properties', 'foo', 'enum']))
        self.assertFalse(e2.context)

    def test_single_nesting(self):
        instance = {'foo': 2, 'bar': [1], 'baz': 15, 'quux': 'spam'}
        schema = {'properties': {'foo': {'type': 'string'}, 'bar': {'minItems': 2}, 'baz': {'maximum': 10, 'enum': [2, 4, 6, 8]}}}
        validator = validators.Draft3Validator(schema)
        errors = validator.iter_errors(instance)
        e1, e2, e3, e4 = sorted_errors(errors)
        self.assertEqual(e1.path, deque(['bar']))
        self.assertEqual(e2.path, deque(['baz']))
        self.assertEqual(e3.path, deque(['baz']))
        self.assertEqual(e4.path, deque(['foo']))
        self.assertEqual(e1.relative_path, deque(['bar']))
        self.assertEqual(e2.relative_path, deque(['baz']))
        self.assertEqual(e3.relative_path, deque(['baz']))
        self.assertEqual(e4.relative_path, deque(['foo']))
        self.assertEqual(e1.absolute_path, deque(['bar']))
        self.assertEqual(e2.absolute_path, deque(['baz']))
        self.assertEqual(e3.absolute_path, deque(['baz']))
        self.assertEqual(e4.absolute_path, deque(['foo']))
        self.assertEqual(e1.json_path, '$.bar')
        self.assertEqual(e2.json_path, '$.baz')
        self.assertEqual(e3.json_path, '$.baz')
        self.assertEqual(e4.json_path, '$.foo')
        self.assertEqual(e1.validator, 'minItems')
        self.assertEqual(e2.validator, 'enum')
        self.assertEqual(e3.validator, 'maximum')
        self.assertEqual(e4.validator, 'type')

    def test_multiple_nesting(self):
        instance = [1, {'foo': 2, 'bar': {'baz': [1]}}, 'quux']
        schema = {'type': 'string', 'items': {'type': ['string', 'object'], 'properties': {'foo': {'enum': [1, 3]}, 'bar': {'type': 'array', 'properties': {'bar': {'required': True}, 'baz': {'minItems': 2}}}}}}
        validator = validators.Draft3Validator(schema)
        errors = validator.iter_errors(instance)
        e1, e2, e3, e4, e5, e6 = sorted_errors(errors)
        self.assertEqual(e1.path, deque([]))
        self.assertEqual(e2.path, deque([0]))
        self.assertEqual(e3.path, deque([1, 'bar']))
        self.assertEqual(e4.path, deque([1, 'bar', 'bar']))
        self.assertEqual(e5.path, deque([1, 'bar', 'baz']))
        self.assertEqual(e6.path, deque([1, 'foo']))
        self.assertEqual(e1.json_path, '$')
        self.assertEqual(e2.json_path, '$[0]')
        self.assertEqual(e3.json_path, '$[1].bar')
        self.assertEqual(e4.json_path, '$[1].bar.bar')
        self.assertEqual(e5.json_path, '$[1].bar.baz')
        self.assertEqual(e6.json_path, '$[1].foo')
        self.assertEqual(e1.schema_path, deque(['type']))
        self.assertEqual(e2.schema_path, deque(['items', 'type']))
        self.assertEqual(list(e3.schema_path), ['items', 'properties', 'bar', 'type'])
        self.assertEqual(list(e4.schema_path), ['items', 'properties', 'bar', 'properties', 'bar', 'required'])
        self.assertEqual(list(e5.schema_path), ['items', 'properties', 'bar', 'properties', 'baz', 'minItems'])
        self.assertEqual(list(e6.schema_path), ['items', 'properties', 'foo', 'enum'])
        self.assertEqual(e1.validator, 'type')
        self.assertEqual(e2.validator, 'type')
        self.assertEqual(e3.validator, 'type')
        self.assertEqual(e4.validator, 'required')
        self.assertEqual(e5.validator, 'minItems')
        self.assertEqual(e6.validator, 'enum')

    def test_recursive(self):
        schema = {'definitions': {'node': {'anyOf': [{'type': 'object', 'required': ['name', 'children'], 'properties': {'name': {'type': 'string'}, 'children': {'type': 'object', 'patternProperties': {'^.*$': {'$ref': '#/definitions/node'}}}}}]}}, 'type': 'object', 'required': ['root'], 'properties': {'root': {'$ref': '#/definitions/node'}}}
        instance = {'root': {'name': 'root', 'children': {'a': {'name': 'a', 'children': {'ab': {'name': 'ab'}}}}}}
        validator = validators.Draft4Validator(schema)
        e, = validator.iter_errors(instance)
        self.assertEqual(e.absolute_path, deque(['root']))
        self.assertEqual(e.absolute_schema_path, deque(['properties', 'root', 'anyOf']))
        self.assertEqual(e.json_path, '$.root')
        e1, = e.context
        self.assertEqual(e1.absolute_path, deque(['root', 'children', 'a']))
        self.assertEqual(e1.absolute_schema_path, deque(['properties', 'root', 'anyOf', 0, 'properties', 'children', 'patternProperties', '^.*$', 'anyOf']))
        self.assertEqual(e1.json_path, '$.root.children.a')
        e2, = e1.context
        self.assertEqual(e2.absolute_path, deque(['root', 'children', 'a', 'children', 'ab']))
        self.assertEqual(e2.absolute_schema_path, deque(['properties', 'root', 'anyOf', 0, 'properties', 'children', 'patternProperties', '^.*$', 'anyOf', 0, 'properties', 'children', 'patternProperties', '^.*$', 'anyOf']))
        self.assertEqual(e2.json_path, '$.root.children.a.children.ab')

    def test_additionalProperties(self):
        instance = {'bar': 'bar', 'foo': 2}
        schema = {'additionalProperties': {'type': 'integer', 'minimum': 5}}
        validator = validators.Draft3Validator(schema)
        errors = validator.iter_errors(instance)
        e1, e2 = sorted_errors(errors)
        self.assertEqual(e1.path, deque(['bar']))
        self.assertEqual(e2.path, deque(['foo']))
        self.assertEqual(e1.json_path, '$.bar')
        self.assertEqual(e2.json_path, '$.foo')
        self.assertEqual(e1.validator, 'type')
        self.assertEqual(e2.validator, 'minimum')

    def test_patternProperties(self):
        instance = {'bar': 1, 'foo': 2}
        schema = {'patternProperties': {'bar': {'type': 'string'}, 'foo': {'minimum': 5}}}
        validator = validators.Draft3Validator(schema)
        errors = validator.iter_errors(instance)
        e1, e2 = sorted_errors(errors)
        self.assertEqual(e1.path, deque(['bar']))
        self.assertEqual(e2.path, deque(['foo']))
        self.assertEqual(e1.json_path, '$.bar')
        self.assertEqual(e2.json_path, '$.foo')
        self.assertEqual(e1.validator, 'type')
        self.assertEqual(e2.validator, 'minimum')

    def test_additionalItems(self):
        instance = ['foo', 1]
        schema = {'items': [], 'additionalItems': {'type': 'integer', 'minimum': 5}}
        validator = validators.Draft3Validator(schema)
        errors = validator.iter_errors(instance)
        e1, e2 = sorted_errors(errors)
        self.assertEqual(e1.path, deque([0]))
        self.assertEqual(e2.path, deque([1]))
        self.assertEqual(e1.json_path, '$[0]')
        self.assertEqual(e2.json_path, '$[1]')
        self.assertEqual(e1.validator, 'type')
        self.assertEqual(e2.validator, 'minimum')

    def test_additionalItems_with_items(self):
        instance = ['foo', 'bar', 1]
        schema = {'items': [{}], 'additionalItems': {'type': 'integer', 'minimum': 5}}
        validator = validators.Draft3Validator(schema)
        errors = validator.iter_errors(instance)
        e1, e2 = sorted_errors(errors)
        self.assertEqual(e1.path, deque([1]))
        self.assertEqual(e2.path, deque([2]))
        self.assertEqual(e1.json_path, '$[1]')
        self.assertEqual(e2.json_path, '$[2]')
        self.assertEqual(e1.validator, 'type')
        self.assertEqual(e2.validator, 'minimum')

    def test_propertyNames(self):
        instance = {'foo': 12}
        schema = {'propertyNames': {'not': {'const': 'foo'}}}
        validator = validators.Draft7Validator(schema)
        error, = validator.iter_errors(instance)
        self.assertEqual(error.validator, 'not')
        self.assertEqual(error.message, "'foo' should not be valid under {'const': 'foo'}")
        self.assertEqual(error.path, deque([]))
        self.assertEqual(error.json_path, '$')
        self.assertEqual(error.schema_path, deque(['propertyNames', 'not']))

    def test_if_then(self):
        schema = {'if': {'const': 12}, 'then': {'const': 13}}
        validator = validators.Draft7Validator(schema)
        error, = validator.iter_errors(12)
        self.assertEqual(error.validator, 'const')
        self.assertEqual(error.message, '13 was expected')
        self.assertEqual(error.path, deque([]))
        self.assertEqual(error.json_path, '$')
        self.assertEqual(error.schema_path, deque(['then', 'const']))

    def test_if_else(self):
        schema = {'if': {'const': 12}, 'else': {'const': 13}}
        validator = validators.Draft7Validator(schema)
        error, = validator.iter_errors(15)
        self.assertEqual(error.validator, 'const')
        self.assertEqual(error.message, '13 was expected')
        self.assertEqual(error.path, deque([]))
        self.assertEqual(error.json_path, '$')
        self.assertEqual(error.schema_path, deque(['else', 'const']))

    def test_boolean_schema_False(self):
        validator = validators.Draft7Validator(False)
        error, = validator.iter_errors(12)
        self.assertEqual((error.message, error.validator, error.validator_value, error.instance, error.schema, error.schema_path, error.json_path), ('False schema does not allow 12', None, None, 12, False, deque([]), '$'))

    def test_ref(self):
        ref, schema = ('someRef', {'additionalProperties': {'type': 'integer'}})
        validator = validators.Draft7Validator({'$ref': ref}, resolver=validators._RefResolver('', {}, store={ref: schema}))
        error, = validator.iter_errors({'foo': 'notAnInteger'})
        self.assertEqual((error.message, error.validator, error.validator_value, error.instance, error.absolute_path, error.schema, error.schema_path, error.json_path), ("'notAnInteger' is not of type 'integer'", 'type', 'integer', 'notAnInteger', deque(['foo']), {'type': 'integer'}, deque(['additionalProperties', 'type']), '$.foo'))

    def test_prefixItems(self):
        schema = {'prefixItems': [{'type': 'string'}, {}, {}, {'maximum': 3}]}
        validator = validators.Draft202012Validator(schema)
        type_error, min_error = validator.iter_errors([1, 2, 'foo', 5])
        self.assertEqual((type_error.message, type_error.validator, type_error.validator_value, type_error.instance, type_error.absolute_path, type_error.schema, type_error.schema_path, type_error.json_path), ("1 is not of type 'string'", 'type', 'string', 1, deque([0]), {'type': 'string'}, deque(['prefixItems', 0, 'type']), '$[0]'))
        self.assertEqual((min_error.message, min_error.validator, min_error.validator_value, min_error.instance, min_error.absolute_path, min_error.schema, min_error.schema_path, min_error.json_path), ('5 is greater than the maximum of 3', 'maximum', 3, 5, deque([3]), {'maximum': 3}, deque(['prefixItems', 3, 'maximum']), '$[3]'))

    def test_prefixItems_with_items(self):
        schema = {'items': {'type': 'string'}, 'prefixItems': [{}]}
        validator = validators.Draft202012Validator(schema)
        e1, e2 = validator.iter_errors(['foo', 2, 'bar', 4, 'baz'])
        self.assertEqual((e1.message, e1.validator, e1.validator_value, e1.instance, e1.absolute_path, e1.schema, e1.schema_path, e1.json_path), ("2 is not of type 'string'", 'type', 'string', 2, deque([1]), {'type': 'string'}, deque(['items', 'type']), '$[1]'))
        self.assertEqual((e2.message, e2.validator, e2.validator_value, e2.instance, e2.absolute_path, e2.schema, e2.schema_path, e2.json_path), ("4 is not of type 'string'", 'type', 'string', 4, deque([3]), {'type': 'string'}, deque(['items', 'type']), '$[3]'))

    def test_contains_too_many(self):
        """
        `contains` + `maxContains` produces only one error, even if there are
        many more incorrectly matching elements.
        """
        schema = {'contains': {'type': 'string'}, 'maxContains': 2}
        validator = validators.Draft202012Validator(schema)
        error, = validator.iter_errors(['foo', 2, 'bar', 4, 'baz', 'quux'])
        self.assertEqual((error.message, error.validator, error.validator_value, error.instance, error.absolute_path, error.schema, error.schema_path, error.json_path), ('Too many items match the given schema (expected at most 2)', 'maxContains', 2, ['foo', 2, 'bar', 4, 'baz', 'quux'], deque([]), {'contains': {'type': 'string'}, 'maxContains': 2}, deque(['contains']), '$'))

    def test_contains_too_few(self):
        schema = {'contains': {'type': 'string'}, 'minContains': 2}
        validator = validators.Draft202012Validator(schema)
        error, = validator.iter_errors(['foo', 2, 4])
        self.assertEqual((error.message, error.validator, error.validator_value, error.instance, error.absolute_path, error.schema, error.schema_path, error.json_path), ('Too few items match the given schema (expected at least 2 but only 1 matched)', 'minContains', 2, ['foo', 2, 4], deque([]), {'contains': {'type': 'string'}, 'minContains': 2}, deque(['contains']), '$'))

    def test_contains_none(self):
        schema = {'contains': {'type': 'string'}, 'minContains': 2}
        validator = validators.Draft202012Validator(schema)
        error, = validator.iter_errors([2, 4])
        self.assertEqual((error.message, error.validator, error.validator_value, error.instance, error.absolute_path, error.schema, error.schema_path, error.json_path), ('[2, 4] does not contain items matching the given schema', 'contains', {'type': 'string'}, [2, 4], deque([]), {'contains': {'type': 'string'}, 'minContains': 2}, deque(['contains']), '$'))

    def test_ref_sibling(self):
        schema = {'$defs': {'foo': {'required': ['bar']}}, 'properties': {'aprop': {'$ref': '#/$defs/foo', 'required': ['baz']}}}
        validator = validators.Draft202012Validator(schema)
        e1, e2 = validator.iter_errors({'aprop': {}})
        self.assertEqual((e1.message, e1.validator, e1.validator_value, e1.instance, e1.absolute_path, e1.schema, e1.schema_path, e1.relative_schema_path, e1.json_path), ("'bar' is a required property", 'required', ['bar'], {}, deque(['aprop']), {'required': ['bar']}, deque(['properties', 'aprop', 'required']), deque(['properties', 'aprop', 'required']), '$.aprop'))
        self.assertEqual((e2.message, e2.validator, e2.validator_value, e2.instance, e2.absolute_path, e2.schema, e2.schema_path, e2.relative_schema_path, e2.json_path), ("'baz' is a required property", 'required', ['baz'], {}, deque(['aprop']), {'$ref': '#/$defs/foo', 'required': ['baz']}, deque(['properties', 'aprop', 'required']), deque(['properties', 'aprop', 'required']), '$.aprop'))