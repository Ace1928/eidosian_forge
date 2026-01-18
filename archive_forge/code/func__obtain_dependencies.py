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
def _obtain_dependencies(self, dist: 'Distribution'):
    if 'dependencies' in self.dynamic:
        value = self._obtain(dist, 'dependencies', {})
        if value:
            return _parse_requirements_list(value)
    return None