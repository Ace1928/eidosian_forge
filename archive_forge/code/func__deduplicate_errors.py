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
def _deduplicate_errors(grouped_errors: GroupedValidationErrors) -> GroupedValidationErrors:
    """Some errors have very similar error messages or are just in general not helpful
    for a user. This function removes as many of these cases as possible and
    can be extended over time to handle new cases that come up.
    """
    grouped_errors_deduplicated: GroupedValidationErrors = {}
    for json_path, element_errors in grouped_errors.items():
        errors_by_validator = _group_errors_by_validator(element_errors)
        deduplication_functions = {'enum': _deduplicate_enum_errors, 'additionalProperties': _deduplicate_additional_properties_errors}
        deduplicated_errors: ValidationErrorList = []
        for validator, errors in errors_by_validator.items():
            deduplication_func = deduplication_functions.get(validator, None)
            if deduplication_func is not None:
                errors = deduplication_func(errors)
            deduplicated_errors.extend(_deduplicate_by_message(errors))
        deduplicated_errors = [err for err in deduplicated_errors if not _is_required_value_error(err)]
        grouped_errors_deduplicated[json_path] = deduplicated_errors
    return grouped_errors_deduplicated