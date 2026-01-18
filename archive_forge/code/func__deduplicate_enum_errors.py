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
def _deduplicate_enum_errors(errors: ValidationErrorList) -> ValidationErrorList:
    """Deduplicate enum errors by removing the errors where the allowed values
    are a subset of another error. For example, if `enum` contains two errors
    and one has `validator_value` (i.e. accepted values) ["A", "B"] and the
    other one ["A", "B", "C"] then the first one is removed and the final
    `enum` list only contains the error with ["A", "B", "C"].
    """
    if len(errors) > 1:
        value_strings = [','.join(err.validator_value) for err in errors]
        longest_enums: ValidationErrorList = []
        for value_str, err in zip(value_strings, errors):
            if not _contained_at_start_of_one_of_other_values(value_str, value_strings):
                longest_enums.append(err)
        errors = longest_enums
    return errors