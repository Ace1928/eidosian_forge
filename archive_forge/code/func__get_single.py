from __future__ import annotations
import os
import re
from typing import Any, Callable, Iterable, TypeVar
from coverage import env
from coverage.exceptions import ConfigError
from coverage.misc import import_third_party, substitute_variables
from coverage.types import TConfigSectionOut, TConfigValueOut
def _get_single(self, section: str, option: str) -> Any:
    """Get a single-valued option.

        Performs environment substitution if the value is a string. Other types
        will be converted later as needed.
        """
    name, value = self._get(section, option)
    if isinstance(value, str):
        value = substitute_variables(value, os.environ)
    return (name, value)