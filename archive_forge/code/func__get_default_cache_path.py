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
def _get_default_cache_path():
    if platform.system().lower() == 'windows':
        dir_ = Path(os.getenv('LOCALAPPDATA') or '~', 'Parso', 'Parso')
    elif platform.system().lower() == 'darwin':
        dir_ = Path('~', 'Library', 'Caches', 'Parso')
    else:
        dir_ = Path(os.getenv('XDG_CACHE_HOME') or '~/.cache', 'parso')
    return dir_.expanduser()