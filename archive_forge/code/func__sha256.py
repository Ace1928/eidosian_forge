import csv
import gzip
import hashlib
import os
import shutil
from collections import namedtuple
from importlib import resources
from numbers import Integral
from os import environ, listdir, makedirs
from os.path import expanduser, isdir, join, splitext
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
from ..preprocessing import scale
from ..utils import Bunch, check_pandas_support, check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, 'rb') as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()