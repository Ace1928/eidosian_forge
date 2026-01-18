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
class TestLatestValidator(TestCase):
    """
    These really apply to multiple versions but are easiest to test on one.
    """

    def test_ref_resolvers_may_have_boolean_schemas_stored(self):
        ref = 'someCoolRef'
        schema = {'$ref': ref}
        resolver = validators._RefResolver('', {}, store={ref: False})
        validator = validators._LATEST_VERSION(schema, resolver=resolver)
        with self.assertRaises(exceptions.ValidationError):
            validator.validate(None)