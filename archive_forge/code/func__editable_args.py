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
def _editable_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
    """
        The ``editable_wheel`` command accepts ``editable-mode=strict``.

        >>> fn = _ConfigSettingsTranslator()._editable_args
        >>> list(fn(None))
        []
        >>> list(fn({"editable-mode": "strict"}))
        ['--mode', 'strict']
        """
    cfg = config_settings or {}
    mode = cfg.get('editable-mode') or cfg.get('editable_mode')
    if not mode:
        return
    yield from ['--mode', str(mode)]