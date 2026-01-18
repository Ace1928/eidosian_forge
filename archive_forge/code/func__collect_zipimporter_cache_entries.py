from glob import glob
from distutils.util import get_platform
from distutils.util import convert_path, subst_vars
from distutils.errors import (
from distutils import log, dir_util
from distutils.command.build_scripts import first_line_re
from distutils.spawn import find_executable
from distutils.command import install
import sys
import os
from typing import Dict, List
import zipimport
import shutil
import tempfile
import zipfile
import re
import stat
import random
import textwrap
import warnings
import site
import struct
import contextlib
import subprocess
import shlex
import io
import configparser
import sysconfig
from sysconfig import get_path
from setuptools import Command
from setuptools.sandbox import run_setup
from setuptools.command import setopt
from setuptools.archive_util import unpack_archive
from setuptools.package_index import (
from setuptools.command import bdist_egg, egg_info
from setuptools.warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
from setuptools.wheel import Wheel
from pkg_resources import (
import pkg_resources
from ..compat import py39, py311
from .._path import ensure_directory
from ..extern.jaraco.text import yield_lines
def _collect_zipimporter_cache_entries(normalized_path, cache):
    """
    Return zipimporter cache entry keys related to a given normalized path.

    Alternative path spellings (e.g. those using different character case or
    those using alternative path separators) related to the same path are
    included. Any sub-path entries are included as well, i.e. those
    corresponding to zip archives embedded in other zip archives.

    """
    result = []
    prefix_len = len(normalized_path)
    for p in cache:
        np = normalize_path(p)
        if np.startswith(normalized_path) and np[prefix_len:prefix_len + 1] in (os.sep, ''):
            result.append(p)
    return result