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
def _subset_to_most_specific_json_paths(errors_by_json_path: GroupedValidationErrors) -> GroupedValidationErrors:
    """Removes key (json path), value (errors) pairs where the json path is fully
    contained in another json path. For example if `errors_by_json_path` has two
    keys, `$.encoding.X` and `$.encoding.X.tooltip`, then the first one will be removed
    and only the second one is returned. This is done under the assumption that
    more specific json paths give more helpful error messages to the user.
    """
    errors_by_json_path_specific: GroupedValidationErrors = {}
    for json_path, errors in errors_by_json_path.items():
        if not _contained_at_start_of_one_of_other_values(json_path, list(errors_by_json_path.keys())):
            errors_by_json_path_specific[json_path] = errors
    return errors_by_json_path_specific