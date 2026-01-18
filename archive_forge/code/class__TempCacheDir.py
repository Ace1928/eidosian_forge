import inspect
import os
import random
import shutil
import tempfile
import weakref
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import xxhash
from . import config
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH
from .utils._dill import dumps
from .utils.deprecation_utils import deprecated
from .utils.logging import get_logger
class _TempCacheDir:
    """
    A temporary directory for storing cached Arrow files with a cleanup that frees references to the Arrow files
    before deleting the directory itself to avoid permission errors on Windows.
    """

    def __init__(self):
        self.name = tempfile.mkdtemp(prefix=config.TEMP_CACHE_DIR_PREFIX)
        self._finalizer = weakref.finalize(self, self._cleanup)

    def _cleanup(self):
        for dset in get_datasets_with_cache_file_in_temp_dir():
            dset.__del__()
        if os.path.exists(self.name):
            try:
                shutil.rmtree(self.name)
            except Exception as e:
                raise OSError(f'An error occured while trying to delete temporary cache directory {self.name}. Please delete it manually.') from e

    def cleanup(self):
        if self._finalizer.detach():
            self._cleanup()