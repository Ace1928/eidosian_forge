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
def _file_with_extension(directory, extension):
    matching = (f for f in os.listdir(directory) if f.endswith(extension))
    try:
        file, = matching
    except ValueError:
        raise ValueError('No distribution was found. Ensure that `setup.py` is not empty and that it calls `setup()`.') from None
    return file