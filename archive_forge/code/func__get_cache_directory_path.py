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
def _get_cache_directory_path(cache_path=None):
    if cache_path is None:
        cache_path = _default_cache_path
    directory = cache_path.joinpath(_VERSION_TAG)
    if not directory.exists():
        os.makedirs(directory)
    return directory