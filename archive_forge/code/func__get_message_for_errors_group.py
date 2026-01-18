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
def _get_message_for_errors_group(self, errors: ValidationErrorList) -> str:
    if errors[0].validator == 'additionalProperties':
        message = self._get_additional_properties_error_message(errors[0])
    else:
        message = self._get_default_error_message(errors=errors)
    return message.strip()