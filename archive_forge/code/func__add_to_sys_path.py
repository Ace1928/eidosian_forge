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
@contextlib.contextmanager
def _add_to_sys_path(path):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)