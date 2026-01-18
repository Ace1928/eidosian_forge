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
def _build_with_temp_dir(self, setup_command, result_extension, result_directory, config_settings):
    result_directory = os.path.abspath(result_directory)
    os.makedirs(result_directory, exist_ok=True)
    temp_opts = {'prefix': '.tmp-', 'dir': result_directory}
    with tempfile.TemporaryDirectory(**temp_opts) as tmp_dist_dir:
        sys.argv = [*sys.argv[:1], *self._global_args(config_settings), *setup_command, '--dist-dir', tmp_dist_dir]
        with no_install_setup_requires():
            self.run_setup()
        result_basename = _file_with_extension(tmp_dist_dir, result_extension)
        result_path = os.path.join(result_directory, result_basename)
        if os.path.exists(result_path):
            os.remove(result_path)
        os.rename(os.path.join(tmp_dist_dir, result_basename), result_path)
    return result_basename