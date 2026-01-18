import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
def _get_section_ref(self, value: Any, *, parent: List[str]=[]) -> Any:
    """Get a single section reference."""
    if isinstance(value, str) and value.startswith(f'"{SECTION_PREFIX}'):
        value = try_load_json(value)
    if isinstance(value, str) and value.startswith(SECTION_PREFIX):
        parts = value.replace(SECTION_PREFIX, '').split('.')
        result = self
        for item in parts:
            try:
                result = result[item]
            except (KeyError, TypeError):
                err_title = 'Error parsing reference to config section'
                err_msg = f"Section '{'.'.join(parts)}' is not defined"
                err = [{'loc': parts, 'msg': err_msg}]
                raise ConfigValidationError(config=self, errors=err, title=err_title) from None
        return result
    elif isinstance(value, str) and SECTION_PREFIX in value:
        err_desc = "Can't reference whole sections or return values of function blocks inside a string or list\n\nYou can change your variable to reference a value instead. Keep in mind that it's not possible to interpolate the return value of a registered function, since variables are interpolated when the config is loaded, and registered functions are resolved afterwards."
        err = [{'loc': parent, 'msg': 'uses section variable in string or list'}]
        raise ConfigValidationError(errors=err, desc=err_desc)
    return value