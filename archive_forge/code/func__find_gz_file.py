from . import hdf5
from .utils import _matrix_to_data_frame
import numpy as np
import os
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import shutil
import tempfile
import urllib
import warnings
import zipfile
def _find_gz_file(*path):
    """Find a file that could be gzipped."""
    path = os.path.join(*path)
    if os.path.isfile(path):
        return path
    else:
        return path + '.gz'