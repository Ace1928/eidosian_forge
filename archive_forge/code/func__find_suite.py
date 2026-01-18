from __future__ import annotations
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json
import os
import re
import subprocess
import sys
import unittest
from attrs import field, frozen
from referencing import Registry
import referencing.jsonschema
from jsonschema.validators import _VALIDATORS
import jsonschema
def _find_suite():
    root = os.environ.get('JSON_SCHEMA_TEST_SUITE')
    if root is not None:
        return Path(root)
    root = Path(jsonschema.__file__).parent.parent / 'json'
    if not root.is_dir():
        raise ValueError("Can't find the JSON-Schema-Test-Suite directory. Set the 'JSON_SCHEMA_TEST_SUITE' environment variable or run the tests from alongside a checkout of the suite.")
    return root