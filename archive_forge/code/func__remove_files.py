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
def _remove_files(self, predicate):
    """
        Remove all files from the file list that match the predicate.
        Return True if any matching files were removed
        """
    found = False
    for i in range(len(self.files) - 1, -1, -1):
        if predicate(self.files[i]):
            self.debug_print(' removing ' + self.files[i])
            del self.files[i]
            found = True
    return found