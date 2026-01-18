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
class ValidatorTestMixin(MetaSchemaTestsMixin):

    def test_it_implements_the_validator_protocol(self):
        self.assertIsInstance(self.Validator({}), protocols.Validator)

    def test_valid_instances_are_valid(self):
        schema, instance = self.valid
        self.assertTrue(self.Validator(schema).is_valid(instance))

    def test_invalid_instances_are_not_valid(self):
        schema, instance = self.invalid
        self.assertFalse(self.Validator(schema).is_valid(instance))

    def test_non_existent_properties_are_ignored(self):
        self.Validator({object(): object()}).validate(instance=object())

    def test_evolve(self):
        schema, format_checker = ({'type': 'integer'}, FormatChecker())
        original = self.Validator(schema, format_checker=format_checker)
        new = original.evolve(schema={'type': 'string'}, format_checker=self.Validator.FORMAT_CHECKER)
        expected = self.Validator({'type': 'string'}, format_checker=self.Validator.FORMAT_CHECKER, _resolver=new._resolver)
        self.assertEqual(new, expected)
        self.assertNotEqual(new, original)

    def test_evolve_with_subclass(self):
        """
        Subclassing validators isn't supported public API, but some users have
        done it, because we don't actually error entirely when it's done :/

        We need to deprecate doing so first to help as many of these users
        ensure they can move to supported APIs, but this test ensures that in
        the interim, we haven't broken those users.
        """
        with self.assertWarns(DeprecationWarning):

            @define
            class OhNo(self.Validator):
                foo = field(factory=lambda: [1, 2, 3])
                _bar = field(default=37)
        validator = OhNo({}, bar=12)
        self.assertEqual(validator.foo, [1, 2, 3])
        new = validator.evolve(schema={'type': 'integer'})
        self.assertEqual(new.foo, [1, 2, 3])
        self.assertEqual(new._bar, 12)

    def test_is_type_is_true_for_valid_type(self):
        self.assertTrue(self.Validator({}).is_type('foo', 'string'))

    def test_is_type_is_false_for_invalid_type(self):
        self.assertFalse(self.Validator({}).is_type('foo', 'array'))

    def test_is_type_evades_bool_inheriting_from_int(self):
        self.assertFalse(self.Validator({}).is_type(True, 'integer'))
        self.assertFalse(self.Validator({}).is_type(True, 'number'))

    def test_it_can_validate_with_decimals(self):
        schema = {'items': {'type': 'number'}}
        Validator = validators.extend(self.Validator, type_checker=self.Validator.TYPE_CHECKER.redefine('number', lambda checker, thing: isinstance(thing, (int, float, Decimal)) and (not isinstance(thing, bool))))
        validator = Validator(schema)
        validator.validate([1, 1.1, Decimal(1) / Decimal(8)])
        invalid = ['foo', {}, [], True, None]
        self.assertEqual([error.instance for error in validator.iter_errors(invalid)], invalid)

    def test_it_returns_true_for_formats_it_does_not_know_about(self):
        validator = self.Validator({'format': 'carrot'}, format_checker=FormatChecker())
        validator.validate('bugs')

    def test_it_does_not_validate_formats_by_default(self):
        validator = self.Validator({})
        self.assertIsNone(validator.format_checker)

    def test_it_validates_formats_if_a_checker_is_provided(self):
        checker = FormatChecker()
        bad = ValueError('Bad!')

        @checker.checks('foo', raises=ValueError)
        def check(value):
            if value == 'good':
                return True
            elif value == 'bad':
                raise bad
            else:
                self.fail(f"What is {value}? [Baby Don't Hurt Me]")
        validator = self.Validator({'format': 'foo'}, format_checker=checker)
        validator.validate('good')
        with self.assertRaises(exceptions.ValidationError) as cm:
            validator.validate('bad')
        self.assertIs(cm.exception.cause, bad)

    def test_non_string_custom_type(self):
        non_string_type = object()
        schema = {'type': [non_string_type]}
        Crazy = validators.extend(self.Validator, type_checker=self.Validator.TYPE_CHECKER.redefine(non_string_type, lambda checker, thing: isinstance(thing, int)))
        Crazy(schema).validate(15)

    def test_it_properly_formats_tuples_in_errors(self):
        """
        A tuple instance properly formats validation errors for uniqueItems.

        See #224
        """
        TupleValidator = validators.extend(self.Validator, type_checker=self.Validator.TYPE_CHECKER.redefine('array', lambda checker, thing: isinstance(thing, tuple)))
        with self.assertRaises(exceptions.ValidationError) as e:
            TupleValidator({'uniqueItems': True}).validate((1, 1))
        self.assertIn('(1, 1) has non-unique elements', str(e.exception))

    def test_check_redefined_sequence(self):
        """
        Allow array to validate against another defined sequence type
        """
        schema = {'type': 'array', 'uniqueItems': True}
        MyMapping = namedtuple('MyMapping', 'a, b')
        Validator = validators.extend(self.Validator, type_checker=self.Validator.TYPE_CHECKER.redefine_many({'array': lambda checker, thing: isinstance(thing, (list, deque)), 'object': lambda checker, thing: isinstance(thing, (dict, MyMapping))}))
        validator = Validator(schema)
        valid_instances = [deque(['a', None, '1', '', True]), deque([[False], [0]]), [deque([False]), deque([0])], [[deque([False])], [deque([0])]], [[[[[deque([False])]]]], [[[[deque([0])]]]]], [deque([deque([False])]), deque([deque([0])])], [MyMapping('a', 0), MyMapping('a', False)], [MyMapping('a', [deque([0])]), MyMapping('a', [deque([False])])], [MyMapping('a', [MyMapping('a', deque([0]))]), MyMapping('a', [MyMapping('a', deque([False]))])], [deque(deque(deque([False]))), deque(deque(deque([0])))]]
        for instance in valid_instances:
            validator.validate(instance)
        invalid_instances = [deque(['a', 'b', 'a']), deque([[False], [False]]), [deque([False]), deque([False])], [[deque([False])], [deque([False])]], [[[[[deque([False])]]]], [[[[deque([False])]]]]], [deque([deque([False])]), deque([deque([False])])], [MyMapping('a', False), MyMapping('a', False)], [MyMapping('a', [deque([False])]), MyMapping('a', [deque([False])])], [MyMapping('a', [MyMapping('a', deque([False]))]), MyMapping('a', [MyMapping('a', deque([False]))])], [deque(deque(deque([False]))), deque(deque(deque([False])))]]
        for instance in invalid_instances:
            with self.assertRaises(exceptions.ValidationError):
                validator.validate(instance)

    def test_it_creates_a_ref_resolver_if_not_provided(self):
        with self.assertWarns(DeprecationWarning):
            resolver = self.Validator({}).resolver
        self.assertIsInstance(resolver, validators._RefResolver)

    def test_it_upconverts_from_deprecated_RefResolvers(self):
        ref, schema = ('someCoolRef', {'type': 'integer'})
        resolver = validators._RefResolver('', {}, store={ref: schema})
        validator = self.Validator({'$ref': ref}, resolver=resolver)
        with self.assertRaises(exceptions.ValidationError):
            validator.validate(None)

    def test_it_upconverts_from_yet_older_deprecated_legacy_RefResolvers(self):
        """
        Legacy RefResolvers support only the context manager form of
        resolution.
        """

        class LegacyRefResolver:

            @contextmanager
            def resolving(this, ref):
                self.assertEqual(ref, 'the ref')
                yield {'type': 'integer'}
        resolver = LegacyRefResolver()
        schema = {'$ref': 'the ref'}
        with self.assertRaises(exceptions.ValidationError):
            self.Validator(schema, resolver=resolver).validate(None)