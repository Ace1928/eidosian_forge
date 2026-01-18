import locale
import logging
import os
import pprint
import re
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict
from typing_extensions import Literal
import numpy as np
def _make_validate_choice_regex(accepted_values, accepted_values_regex, allow_none=False):
    """Validate value is in accepted_values with regex.

    Parameters
    ----------
    accepted_values : iterable
        Iterable containing all accepted_values.
    accepted_values_regex : iterable
        Iterable containing all accepted_values with regex string.
    allow_none: boolean, optional
        Whether to accept ``None`` in addition to the values in ``accepted_values``.
    typeof: type, optional
        Type the values should be converted to.
    """

    def validate_choice_regex(value):
        if allow_none and (value is None or (isinstance(value, str) and value.lower() == 'none')):
            return None
        value = str(value)
        if isinstance(value, str):
            value = value.lower()
        if value in accepted_values:
            value = {'true': True, 'false': False}.get(value, value)
            return value
        elif any((re.match(pattern, value) for pattern in accepted_values_regex)):
            return value
        raise ValueError(f'{value} is not one of {accepted_values} or in regex {accepted_values_regex}{(' nor None' if allow_none else '')}')
    return validate_choice_regex