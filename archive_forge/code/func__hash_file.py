from abc import abstractmethod
from contextlib import closing
import functools
import hashlib
import multiprocessing
import multiprocessing.dummy
import os
import queue
import random
import shutil
import sys  # pylint: disable=unused-import
import tarfile
import threading
import time
import typing
import urllib
import weakref
import zipfile
import numpy as np
from tensorflow.python.framework import tensor
from six.moves.urllib.request import urlopen
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

  Example:

  ```python
  _hash_file('/path/to/file.zip')
  'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
  ```

  Args:
      fpath: path to the file being validated
      algorithm: hash algorithm, one of `'auto'`, `'sha256'`, or `'md5'`.
          The default `'auto'` detects the hash algorithm in use.
      chunk_size: Bytes to read at a time, important for large files.

  Returns:
      The file hash
  """
    if isinstance(algorithm, str):
        hasher = _resolve_hasher(algorithm)
    else:
        hasher = algorithm
    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()