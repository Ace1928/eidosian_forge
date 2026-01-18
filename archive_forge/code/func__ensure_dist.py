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
def _ensure_dist(self) -> 'Distribution':
    from setuptools.dist import Distribution
    attrs = {'src_root': self.root_dir, 'name': self.project_cfg.get('name', None)}
    return self._dist or Distribution(attrs)