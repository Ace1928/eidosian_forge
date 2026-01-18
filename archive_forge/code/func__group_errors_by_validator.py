import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
def _group_errors_by_validator(errors: ValidationErrorList) -> GroupedValidationErrors:
    """Groups the errors by the json schema "validator" that casued the error. For
    example if the error is that a value is not one of an enumeration in the json schema
    then the "validator" is `"enum"`, if the error is due to an unknown property that
    was set although no additional properties are allowed then "validator" is
    `"additionalProperties`, etc.
    """
    errors_by_validator: DefaultDict[str, ValidationErrorList] = collections.defaultdict(list)
    for err in errors:
        errors_by_validator[err.validator].append(err)
    return dict(errors_by_validator)