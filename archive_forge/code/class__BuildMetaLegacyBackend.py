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
class _BuildMetaLegacyBackend(_BuildMetaBackend):
    """Compatibility backend for setuptools

    This is a version of setuptools.build_meta that endeavors
    to maintain backwards
    compatibility with pre-PEP 517 modes of invocation. It
    exists as a temporary
    bridge between the old packaging mechanism and the new
    packaging mechanism,
    and will eventually be removed.
    """

    def run_setup(self, setup_script='setup.py'):
        sys_path = list(sys.path)
        script_dir = os.path.dirname(os.path.abspath(setup_script))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        sys_argv_0 = sys.argv[0]
        sys.argv[0] = setup_script
        try:
            super().run_setup(setup_script=setup_script)
        finally:
            sys.path[:] = sys_path
            sys.argv[0] = sys_argv_0