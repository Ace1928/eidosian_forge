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
def _update_zipimporter_cache(normalized_path, cache, updater=None):
    """
    Update zipimporter cache data for a given normalized path.

    Any sub-path entries are processed as well, i.e. those corresponding to zip
    archives embedded in other zip archives.

    Given updater is a callable taking a cache entry key and the original entry
    (after already removing the entry from the cache), and expected to update
    the entry and possibly return a new one to be inserted in its place.
    Returning None indicates that the entry should not be replaced with a new
    one. If no updater is given, the cache entries are simply removed without
    any additional processing, the same as if the updater simply returned None.

    """
    for p in _collect_zipimporter_cache_entries(normalized_path, cache):
        old_entry = cache[p]
        del cache[p]
        new_entry = updater and updater(p, old_entry)
        if new_entry is not None:
            cache[p] = new_entry