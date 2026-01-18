import time
import os
import sys
import hashlib
import gc
import shutil
import platform
import logging
import warnings
import pickle
from pathlib import Path
from typing import Dict, Any
def _remove_cache_and_update_lock(cache_path=None):
    lock_path = _get_cache_clear_lock_path(cache_path=cache_path)
    try:
        clear_lock_time = os.path.getmtime(lock_path)
    except FileNotFoundError:
        clear_lock_time = None
    if clear_lock_time is None or clear_lock_time + _CACHE_CLEAR_THRESHOLD <= time.time():
        if not _touch(lock_path):
            return False
        clear_inactive_cache(cache_path=cache_path)