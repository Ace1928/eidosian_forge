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
def _get_cache_clear_lock_path(cache_path=None):
    """
    The path where the cache lock is stored.

    Cache lock will prevent continous cache clearing and only allow garbage
    collection once a day (can be configured in _CACHE_CLEAR_THRESHOLD).
    """
    cache_path = cache_path or _default_cache_path
    return cache_path.joinpath('PARSO-CACHE-LOCK')