import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
def _nthreads_update(self):
    """Update environment with new number of threads."""
    self.inputs.environ['OMP_NUM_THREADS'] = '%d' % self.inputs.num_threads