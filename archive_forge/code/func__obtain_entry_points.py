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
def _obtain_entry_points(self, dist: 'Distribution', package_dir: Mapping[str, str]) -> Optional[Dict[str, dict]]:
    fields = ('entry-points', 'scripts', 'gui-scripts')
    if not any((field in self.dynamic for field in fields)):
        return None
    text = self._obtain(dist, 'entry-points', package_dir)
    if text is None:
        return None
    groups = _expand.entry_points(text)
    expanded = {'entry-points': groups}

    def _set_scripts(field: str, group: str):
        if group in groups:
            value = groups.pop(group)
            if field not in self.dynamic:
                raise InvalidConfigError(_MissingDynamic.details(field, value))
            expanded[field] = value
    _set_scripts('scripts', 'console_scripts')
    _set_scripts('gui-scripts', 'gui_scripts')
    return expanded