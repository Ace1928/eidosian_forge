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
def _get_dist_info_dir(self, metadata_directory: Optional[str]) -> Optional[str]:
    if not metadata_directory:
        return None
    dist_info_candidates = list(Path(metadata_directory).glob('*.dist-info'))
    assert len(dist_info_candidates) <= 1
    return str(dist_info_candidates[0]) if dist_info_candidates else None