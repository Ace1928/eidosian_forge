import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Optional, Set, Union
from ..errors import FileError, InvalidConfigError
from ..warnings import SetuptoolsWarning
from . import expand as _expand
from ._apply_pyprojecttoml import _PREVIOUSLY_DEFINED, _MissingDynamic
from ._apply_pyprojecttoml import apply as _apply
def _set_scripts(field: str, group: str):
    if group in groups:
        value = groups.pop(group)
        if field not in self.dynamic:
            raise InvalidConfigError(_MissingDynamic.details(field, value))
        expanded[field] = value