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
def _process_field(self, container: dict, field: str, fn: Callable):
    if field in container:
        with _ignore_errors(self.ignore_option_errors):
            container[field] = fn(container[field])