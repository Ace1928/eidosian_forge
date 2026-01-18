import errno
import logging
import os
from gzip import GzipFile
from os.path import exists, join
import joblib
import numpy as np
from ..utils import Bunch, check_random_state
from ..utils import shuffle as shuffle_method
from ..utils._param_validation import StrOptions, validate_params
from . import get_data_home
from ._base import (
def _mkdirp(d):
    """Ensure directory d exists (like mkdir -p on Unix)
    No guarantee that the directory is writable.
    """
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise