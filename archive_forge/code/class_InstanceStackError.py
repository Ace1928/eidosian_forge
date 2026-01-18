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
class InstanceStackError(DFTError):
    """bad series of instance numbers"""

    def __init__(self, series, i, si):
        self.series = series
        self.i = i
        self.si = si

    def __str__(self):
        fmt = 'expecting instance number %d, got %d'
        return fmt % (self.i + 1, self.si.instance_number)