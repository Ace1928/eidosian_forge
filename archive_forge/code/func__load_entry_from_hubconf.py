import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import torch
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Optional, Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request
from urllib.parse import urlparse  # noqa: F401
from torch.serialization import MAP_LOCATION
def _load_entry_from_hubconf(m, model):
    if not isinstance(model, str):
        raise ValueError('Invalid input: model should be a string of function name')
    _check_dependencies(m)
    func = _load_attr_from_module(m, model)
    if func is None or not callable(func):
        raise RuntimeError(f'Cannot find callable {model} in hubconf')
    return func