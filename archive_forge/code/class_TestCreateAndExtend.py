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
class TestCreateAndExtend(TestCase):

    def setUp(self):
        self.addCleanup(self.assertEqual, validators._META_SCHEMAS, dict(validators._META_SCHEMAS))
        self.addCleanup(self.assertEqual, validators._VALIDATORS, dict(validators._VALIDATORS))
        self.meta_schema = {'$id': 'some://meta/schema'}
        self.validators = {'fail': fail}
        self.type_checker = TypeChecker()
        self.Validator = validators.create(meta_schema=self.meta_schema, validators=self.validators, type_checker=self.type_checker)

    def test_attrs(self):
        self.assertEqual((self.Validator.VALIDATORS, self.Validator.META_SCHEMA, self.Validator.TYPE_CHECKER), (self.validators, self.meta_schema, self.type_checker))

    def test_init(self):
        schema = {'fail': []}
        self.assertEqual(self.Validator(schema).schema, schema)

    def test_iter_errors_successful(self):
        schema = {'fail': []}
        validator = self.Validator(schema)
        errors = list(validator.iter_errors('hello'))
        self.assertEqual(errors, [])

    def test_iter_errors_one_error(self):
        schema = {'fail': [{'message': 'Whoops!'}]}
        validator = self.Validator(schema)
        expected_error = exceptions.ValidationError('Whoops!', instance='goodbye', schema=schema, validator='fail', validator_value=[{'message': 'Whoops!'}], schema_path=deque(['fail']))
        errors = list(validator.iter_errors('goodbye'))
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]._contents(), expected_error._contents())

    def test_iter_errors_multiple_errors(self):
        schema = {'fail': [{'message': 'First'}, {'message': 'Second!', 'validator': 'asdf'}, {'message': 'Third'}]}
        validator = self.Validator(schema)
        errors = list(validator.iter_errors('goodbye'))
        self.assertEqual(len(errors), 3)

    def test_if_a_version_is_provided_it_is_registered(self):
        Validator = validators.create(meta_schema={'$id': 'something'}, version='my version')
        self.addCleanup(validators._META_SCHEMAS.pop, 'something')
        self.addCleanup(validators._VALIDATORS.pop, 'my version')
        self.assertEqual(Validator.__name__, 'MyVersionValidator')
        self.assertEqual(Validator.__qualname__, 'MyVersionValidator')

    def test_repr(self):
        Validator = validators.create(meta_schema={'$id': 'something'}, version='my version')
        self.addCleanup(validators._META_SCHEMAS.pop, 'something')
        self.addCleanup(validators._VALIDATORS.pop, 'my version')
        self.assertEqual(repr(Validator({})), 'MyVersionValidator(schema={}, format_checker=None)')

    def test_long_repr(self):
        Validator = validators.create(meta_schema={'$id': 'something'}, version='my version')
        self.addCleanup(validators._META_SCHEMAS.pop, 'something')
        self.addCleanup(validators._VALIDATORS.pop, 'my version')
        self.assertEqual(repr(Validator({'a': list(range(1000))})), "MyVersionValidator(schema={'a': [0, 1, 2, 3, 4, 5, ...]}, format_checker=None)")

    def test_repr_no_version(self):
        Validator = validators.create(meta_schema={})
        self.assertEqual(repr(Validator({})), 'Validator(schema={}, format_checker=None)')

    def test_dashes_are_stripped_from_validator_names(self):
        Validator = validators.create(meta_schema={'$id': 'something'}, version='foo-bar')
        self.addCleanup(validators._META_SCHEMAS.pop, 'something')
        self.addCleanup(validators._VALIDATORS.pop, 'foo-bar')
        self.assertEqual(Validator.__qualname__, 'FooBarValidator')

    def test_if_a_version_is_not_provided_it_is_not_registered(self):
        original = dict(validators._META_SCHEMAS)
        validators.create(meta_schema={'id': 'id'})
        self.assertEqual(validators._META_SCHEMAS, original)

    def test_validates_registers_meta_schema_id(self):
        meta_schema_key = 'meta schema id'
        my_meta_schema = {'id': meta_schema_key}
        validators.create(meta_schema=my_meta_schema, version='my version', id_of=lambda s: s.get('id', ''))
        self.addCleanup(validators._META_SCHEMAS.pop, meta_schema_key)
        self.addCleanup(validators._VALIDATORS.pop, 'my version')
        self.assertIn(meta_schema_key, validators._META_SCHEMAS)

    def test_validates_registers_meta_schema_draft6_id(self):
        meta_schema_key = 'meta schema $id'
        my_meta_schema = {'$id': meta_schema_key}
        validators.create(meta_schema=my_meta_schema, version='my version')
        self.addCleanup(validators._META_SCHEMAS.pop, meta_schema_key)
        self.addCleanup(validators._VALIDATORS.pop, 'my version')
        self.assertIn(meta_schema_key, validators._META_SCHEMAS)

    def test_create_default_types(self):
        Validator = validators.create(meta_schema={}, validators=())
        self.assertTrue(all((Validator({}).is_type(instance=instance, type=type) for type, instance in [('array', []), ('boolean', True), ('integer', 12), ('null', None), ('number', 12.0), ('object', {}), ('string', 'foo')])))

    def test_check_schema_with_different_metaschema(self):
        """
        One can create a validator class whose metaschema uses a different
        dialect than itself.
        """
        NoEmptySchemasValidator = validators.create(meta_schema={'$schema': validators.Draft202012Validator.META_SCHEMA['$id'], 'not': {'const': {}}})
        NoEmptySchemasValidator.check_schema({'foo': 'bar'})
        with self.assertRaises(exceptions.SchemaError):
            NoEmptySchemasValidator.check_schema({})
        NoEmptySchemasValidator({'foo': 'bar'}).validate('foo')

    def test_check_schema_with_different_metaschema_defaults_to_self(self):
        """
        A validator whose metaschema doesn't declare $schema defaults to its
        own validation behavior, not the latest "normal" specification.
        """
        NoEmptySchemasValidator = validators.create(meta_schema={'fail': [{'message': 'Meta schema whoops!'}]}, validators={'fail': fail})
        with self.assertRaises(exceptions.SchemaError):
            NoEmptySchemasValidator.check_schema({})

    def test_extend(self):
        original = dict(self.Validator.VALIDATORS)
        new = object()
        Extended = validators.extend(self.Validator, validators={'new': new})
        self.assertEqual((Extended.VALIDATORS, Extended.META_SCHEMA, Extended.TYPE_CHECKER, self.Validator.VALIDATORS), (dict(original, new=new), self.Validator.META_SCHEMA, self.Validator.TYPE_CHECKER, original))

    def test_extend_idof(self):
        """
        Extending a validator preserves its notion of schema IDs.
        """

        def id_of(schema):
            return schema.get('__test__', self.Validator.ID_OF(schema))
        correct_id = 'the://correct/id/'
        meta_schema = {'$id': 'the://wrong/id/', '__test__': correct_id}
        Original = validators.create(meta_schema=meta_schema, validators=self.validators, type_checker=self.type_checker, id_of=id_of)
        self.assertEqual(Original.ID_OF(Original.META_SCHEMA), correct_id)
        Derived = validators.extend(Original)
        self.assertEqual(Derived.ID_OF(Derived.META_SCHEMA), correct_id)

    def test_extend_applicable_validators(self):
        """
        Extending a validator preserves its notion of applicable validators.
        """
        schema = {'$defs': {'test': {'type': 'number'}}, '$ref': '#/$defs/test', 'maximum': 1}
        draft4 = validators.Draft4Validator(schema)
        self.assertTrue(draft4.is_valid(37))
        Derived = validators.extend(validators.Draft4Validator)
        self.assertTrue(Derived(schema).is_valid(37))