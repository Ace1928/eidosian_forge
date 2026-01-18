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
def _obtain_optional_dependencies(self, dist: 'Distribution'):
    if 'optional-dependencies' not in self.dynamic:
        return None
    if 'optional-dependencies' in self.dynamic_cfg:
        optional_dependencies_map = self.dynamic_cfg['optional-dependencies']
        assert isinstance(optional_dependencies_map, dict)
        return {group: _parse_requirements_list(self._expand_directive(f'tool.setuptools.dynamic.optional-dependencies.{group}', directive, {})) for group, directive in optional_dependencies_map.items()}
    self._ensure_previously_set(dist, 'optional-dependencies')
    return None