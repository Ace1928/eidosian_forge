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
def _obtain(self, dist: 'Distribution', field: str, package_dir: Mapping[str, str]):
    if field in self.dynamic_cfg:
        return self._expand_directive(f'tool.setuptools.dynamic.{field}', self.dynamic_cfg[field], package_dir)
    self._ensure_previously_set(dist, field)
    return None