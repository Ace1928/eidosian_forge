import contextlib
import getpass
import logging
import os
import sqlite3
import tempfile
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy
from nibabel.optpkg import optional_package
from .nifti1 import Nifti1Header
def _get_subdirs(base_dir, files_dict=None, followlinks=False):
    dirs = []
    for dirpath, dirnames, filenames in os.walk(base_dir, followlinks=followlinks):
        abs_dir = os.path.realpath(dirpath)
        if abs_dir in dirs:
            raise CachingError(f'link cycle detected under {base_dir}')
        dirs.append(abs_dir)
        if files_dict is not None:
            files_dict[abs_dir] = filenames
    return dirs