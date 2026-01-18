import abc
import hashlib
import os
import tempfile
from pathlib import Path
from ..common.build import _build
from .cache import get_cache_manager
def __delattr__(self, name):
    self._initialize_obj()
    delattr(self._obj, name)