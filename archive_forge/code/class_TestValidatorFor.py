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
class TestValidatorFor(TestCase):

    def test_draft_3(self):
        schema = {'$schema': 'http://json-schema.org/draft-03/schema'}
        self.assertIs(validators.validator_for(schema), validators.Draft3Validator)
        schema = {'$schema': 'http://json-schema.org/draft-03/schema#'}
        self.assertIs(validators.validator_for(schema), validators.Draft3Validator)

    def test_draft_4(self):
        schema = {'$schema': 'http://json-schema.org/draft-04/schema'}
        self.assertIs(validators.validator_for(schema), validators.Draft4Validator)
        schema = {'$schema': 'http://json-schema.org/draft-04/schema#'}
        self.assertIs(validators.validator_for(schema), validators.Draft4Validator)

    def test_draft_6(self):
        schema = {'$schema': 'http://json-schema.org/draft-06/schema'}
        self.assertIs(validators.validator_for(schema), validators.Draft6Validator)
        schema = {'$schema': 'http://json-schema.org/draft-06/schema#'}
        self.assertIs(validators.validator_for(schema), validators.Draft6Validator)

    def test_draft_7(self):
        schema = {'$schema': 'http://json-schema.org/draft-07/schema'}
        self.assertIs(validators.validator_for(schema), validators.Draft7Validator)
        schema = {'$schema': 'http://json-schema.org/draft-07/schema#'}
        self.assertIs(validators.validator_for(schema), validators.Draft7Validator)

    def test_draft_201909(self):
        schema = {'$schema': 'https://json-schema.org/draft/2019-09/schema'}
        self.assertIs(validators.validator_for(schema), validators.Draft201909Validator)
        schema = {'$schema': 'https://json-schema.org/draft/2019-09/schema#'}
        self.assertIs(validators.validator_for(schema), validators.Draft201909Validator)

    def test_draft_202012(self):
        schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema'}
        self.assertIs(validators.validator_for(schema), validators.Draft202012Validator)
        schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema#'}
        self.assertIs(validators.validator_for(schema), validators.Draft202012Validator)

    def test_True(self):
        self.assertIs(validators.validator_for(True), validators._LATEST_VERSION)

    def test_False(self):
        self.assertIs(validators.validator_for(False), validators._LATEST_VERSION)

    def test_custom_validator(self):
        Validator = validators.create(meta_schema={'id': 'meta schema id'}, version='12', id_of=lambda s: s.get('id', ''))
        schema = {'$schema': 'meta schema id'}
        self.assertIs(validators.validator_for(schema), Validator)

    def test_custom_validator_draft6(self):
        Validator = validators.create(meta_schema={'$id': 'meta schema $id'}, version='13')
        schema = {'$schema': 'meta schema $id'}
        self.assertIs(validators.validator_for(schema), Validator)

    def test_validator_for_jsonschema_default(self):
        self.assertIs(validators.validator_for({}), validators._LATEST_VERSION)

    def test_validator_for_custom_default(self):
        self.assertIs(validators.validator_for({}, default=None), None)

    def test_warns_if_meta_schema_specified_was_not_found(self):
        with self.assertWarns(DeprecationWarning) as cm:
            validators.validator_for(schema={'$schema': 'unknownSchema'})
        self.assertEqual(cm.filename, __file__)
        self.assertEqual(str(cm.warning), 'The metaschema specified by $schema was not found. Using the latest draft to validate, but this will raise an error in the future.')

    def test_does_not_warn_if_meta_schema_is_unspecified(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            validators.validator_for(schema={}, default={})
        self.assertFalse(w)

    def test_validator_for_custom_default_with_schema(self):
        schema, default = ({'$schema': 'mailto:foo@example.com'}, object())
        self.assertIs(validators.validator_for(schema, default), default)