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
def _save_to_file_system(hashed_grammar, path, item, cache_path=None):
    with open(_get_hashed_path(hashed_grammar, path, cache_path=cache_path), 'wb') as f:
        pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)