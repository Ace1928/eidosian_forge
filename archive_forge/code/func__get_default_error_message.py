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
def _get_default_error_message(self, errors: ValidationErrorList) -> str:
    bullet_points: List[str] = []
    errors_by_validator = _group_errors_by_validator(errors)
    if 'enum' in errors_by_validator:
        for error in errors_by_validator['enum']:
            bullet_points.append(f'one of {error.validator_value}')
    if 'type' in errors_by_validator:
        types = [f"'{err.validator_value}'" for err in errors_by_validator['type']]
        point = 'of type '
        if len(types) == 1:
            point += types[0]
        elif len(types) == 2:
            point += f'{types[0]} or {types[1]}'
        else:
            point += ', '.join(types[:-1]) + f', or {types[-1]}'
        bullet_points.append(point)
    error = errors[0]
    message = f"'{error.instance}' is an invalid value"
    if error.absolute_path:
        message += f' for `{error.absolute_path[-1]}`'
    if len(bullet_points) == 0:
        message += '.\n\n'
    elif len(bullet_points) == 1:
        message += f'. Valid values are {bullet_points[0]}.\n\n'
    else:
        bullet_points = [point[0].upper() + point[1:] for point in bullet_points]
        message += '. Valid values are:\n\n'
        message += '\n'.join([f'- {point}' for point in bullet_points])
        message += '\n\n'
    for validator, errors in errors_by_validator.items():
        if validator not in ('enum', 'type'):
            message += '\n'.join([e.message for e in errors])
    return message