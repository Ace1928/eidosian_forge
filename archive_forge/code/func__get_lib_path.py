import importlib
import logging
import os
import types
from pathlib import Path
import torch
def _get_lib_path(lib: str):
    suffix = 'pyd' if os.name == 'nt' else 'so'
    path = _LIB_DIR / f'{lib}.{suffix}'
    return path