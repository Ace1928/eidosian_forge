import hashlib
import logging
import numpy as np
import os
import pandas as pd
import tarfile
import tempfile
import six
import shutil
from .core import PATH_TYPES, fspath
def _calc_md5(path):
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()