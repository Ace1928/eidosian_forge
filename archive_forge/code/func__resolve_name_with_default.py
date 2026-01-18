from importlib import metadata
from json import JSONDecodeError
from textwrap import dedent
import argparse
import json
import sys
import traceback
import warnings
from attrs import define, field
from jsonschema.exceptions import SchemaError
from jsonschema.validators import _RefResolver, validator_for
def _resolve_name_with_default(name):
    if '.' not in name:
        name = 'jsonschema.' + name
    return resolve_name(name)