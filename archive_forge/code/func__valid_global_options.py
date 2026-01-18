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
def _valid_global_options(self):
    """Global options accepted by setuptools (e.g. quiet or verbose)."""
    options = (opt[:2] for opt in setuptools.dist.Distribution.global_options)
    return {flag for long_and_short in options for flag in long_and_short if flag}