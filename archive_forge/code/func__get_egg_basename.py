from distutils.filelist import FileList as _FileList
from distutils.errors import DistutilsInternalError
from distutils.util import convert_path
from distutils import log
import distutils.errors
import distutils.filelist
import functools
import os
import re
import sys
import time
import collections
from .._importlib import metadata
from .. import _entry_points, _normalization
from . import _requirestxt
from setuptools import Command
from setuptools.command.sdist import sdist
from setuptools.command.sdist import walk_revctrl
from setuptools.command.setopt import edit_config
from setuptools.command import bdist_egg
import setuptools.unicode_utils as unicode_utils
from setuptools.glob import glob
from setuptools.extern import packaging
from ..warnings import SetuptoolsDeprecationWarning
def _get_egg_basename(self, py_version=PY_MAJOR, platform=None):
    """Compute filename of the output egg. Private API."""
    return _egg_basename(self.egg_name, self.egg_version, py_version, platform)