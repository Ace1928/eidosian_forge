import io
import os
import shlex
import sys
import tokenize
import shutil
import contextlib
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
import setuptools
import distutils
from . import errors
from ._path import same_path
from ._reqs import parse_strings
from .warnings import SetuptoolsDeprecationWarning
from distutils.util import strtobool
def __dist_info_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
    """
        The ``dist_info`` command accepts ``tag-date`` and ``tag-build``.

        .. warning::
           We cannot use this yet as it requires the ``sdist`` and ``bdist_wheel``
           commands run in ``build_sdist`` and ``build_wheel`` to reuse the egg-info
           directory created in ``prepare_metadata_for_build_wheel``.

        >>> fn = _ConfigSettingsTranslator()._ConfigSettingsTranslator__dist_info_args
        >>> list(fn(None))
        []
        >>> list(fn({"tag-date": "False"}))
        ['--no-date']
        >>> list(fn({"tag-date": None}))
        ['--no-date']
        >>> list(fn({"tag-date": "true", "tag-build": ".a"}))
        ['--tag-date', '--tag-build', '.a']
        """
    cfg = config_settings or {}
    if 'tag-date' in cfg:
        val = strtobool(str(cfg['tag-date'] or 'false'))
        yield ('--tag-date' if val else '--no-date')
    if 'tag-build' in cfg:
        yield from ['--tag-build', str(cfg['tag-build'])]