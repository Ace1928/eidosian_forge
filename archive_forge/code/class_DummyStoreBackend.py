import functools
import gc
import logging
import shutil
import os
import os.path
import pathlib
import pickle
import sys
import time
import datetime
import textwrap
import pytest
from joblib.memory import Memory
from joblib.memory import expires_after
from joblib.memory import MemorizedFunc, NotMemorizedFunc
from joblib.memory import MemorizedResult, NotMemorizedResult
from joblib.memory import _FUNCTION_HASHES
from joblib.memory import register_store_backend, _STORE_BACKENDS
from joblib.memory import _build_func_identifier, _store_backend_factory
from joblib.memory import JobLibCollisionWarning
from joblib.parallel import Parallel, delayed
from joblib._store_backends import StoreBackendBase, FileSystemStoreBackend
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.testing import parametrize, raises, warns
from joblib.hashing import hash
class DummyStoreBackend(StoreBackendBase):
    """A dummy store backend that does nothing."""

    def _open_item(self, *args, **kwargs):
        """Open an item on store."""
        'Does nothing'

    def _item_exists(self, location):
        """Check if an item location exists."""
        'Does nothing'

    def _move_item(self, src, dst):
        """Move an item from src to dst in store."""
        'Does nothing'

    def create_location(self, location):
        """Create location on store."""
        'Does nothing'

    def exists(self, obj):
        """Check if an object exists in the store"""
        return False

    def clear_location(self, obj):
        """Clear object on store"""
        'Does nothing'

    def get_items(self):
        """Returns the whole list of items available in cache."""
        return []

    def configure(self, location, *args, **kwargs):
        """Configure the store"""
        'Does nothing'