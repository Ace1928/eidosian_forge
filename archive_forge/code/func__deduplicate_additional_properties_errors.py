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
def _deduplicate_additional_properties_errors(errors: ValidationErrorList) -> ValidationErrorList:
    """If there are multiple additional property errors it usually means that
    the offending element was validated against multiple schemas and
    its parent is a common anyOf validator.
    The error messages produced from these cases are usually
    very similar and we just take the shortest one. For example,
    the following 3 errors are raised for the `unknown` channel option in
    `alt.X("variety", unknown=2)`:
    - "Additional properties are not allowed ('unknown' was unexpected)"
    - "Additional properties are not allowed ('field', 'unknown' were unexpected)"
    - "Additional properties are not allowed ('field', 'type', 'unknown' were unexpected)"
    """
    if len(errors) > 1:
        parent = errors[0].parent
        if parent is not None and parent.validator == 'anyOf' and all((err.parent is parent for err in errors[1:])):
            errors = [min(errors, key=lambda x: len(x.message))]
    return errors