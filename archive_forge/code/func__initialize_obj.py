import abc
import hashlib
import os
import tempfile
from pathlib import Path
from ..common.build import _build
from .cache import get_cache_manager
def _initialize_obj(self):
    if self._obj is None:
        self._obj = self._init_fn()