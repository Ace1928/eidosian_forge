from __future__ import annotations
import io
import logging
import os
import threading
import warnings
import weakref
from errno import ESPIPE
from glob import has_magic
from hashlib import sha256
from typing import ClassVar
from .callbacks import DEFAULT_CALLBACK
from .config import apply_config, conf
from .dircache import DirCache
from .transaction import Transaction
from .utils import (
def _get_pyarrow_filesystem(self):
    """
        Make a version of the FS instance which will be acceptable to pyarrow
        """
    return self