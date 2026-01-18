import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
class StubO:
    """Simple file-like object that allows writes with any type and records."""

    def __init__(self):
        self.write_record = []

    def write(self, data):
        self.write_record.append(data)

    def check_types(self, testcase, expected_type):
        testcase.assertFalse(any((not isinstance(o, expected_type) for o in self.write_record)), 'Not all writes of type {}: {!r}'.format(expected_type.__name__, self.write_record))