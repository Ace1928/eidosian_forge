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

        PEP 517 requires that the .dist-info directory be placed in the
        metadata_directory. To comply, we MUST copy the directory to the root.

        Returns the basename of the info directory, e.g. `proj-0.0.0.dist-info`.
        