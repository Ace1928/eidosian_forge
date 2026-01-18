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
def _expand_directive(self, specifier: str, directive, package_dir: Mapping[str, str]):
    from setuptools.extern.more_itertools import always_iterable
    with _ignore_errors(self.ignore_option_errors):
        root_dir = self.root_dir
        if 'file' in directive:
            self._referenced_files.update(always_iterable(directive['file']))
            return _expand.read_files(directive['file'], root_dir)
        if 'attr' in directive:
            return _expand.read_attr(directive['attr'], package_dir, root_dir)
        raise ValueError(f'invalid `{specifier}`: {directive!r}')
    return None