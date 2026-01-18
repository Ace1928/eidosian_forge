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
def _obtain_version(self, dist: 'Distribution', package_dir: Mapping[str, str]):
    if 'version' in self.dynamic and 'version' in self.dynamic_cfg:
        return _expand.version(self._obtain(dist, 'version', package_dir))
    return None