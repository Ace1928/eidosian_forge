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
def _ensure_previously_set(self, dist: 'Distribution', field: str):
    previous = _PREVIOUSLY_DEFINED[field](dist)
    if previous is None and (not self.ignore_option_errors):
        msg = f'No configuration found for dynamic {field!r}.\nSome dynamic fields need to be specified via `tool.setuptools.dynamic`\nothers must be specified via the equivalent attribute in `setup.py`.'
        raise InvalidConfigError(msg)