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
def _validate_dict_of_lists(values):
    if isinstance(values, dict):
        return {key: tuple(item) for key, item in values.items()}
    validated_dict = {}
    for value in values:
        tup = value.split(':', 1)
        if len(tup) != 2:
            raise ValueError(f"Could not interpret '{value}' as key: list or str")
        key, vals = tup
        key = key.strip(' "')
        vals = [val.strip(' "') for val in vals.strip(' [],').split(',')]
        if key in validated_dict:
            warnings.warn(f'Repeated key {key} when validating dict of lists')
        validated_dict[key] = tuple(vals)
    return validated_dict