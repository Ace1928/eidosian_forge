import bz2
import gzip
import lzma
import os
import shutil
import struct
import tarfile
import warnings
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union
from .. import config
from ._filelock import FileLock
from .logging import get_logger
def _get_output_path(self, path: str) -> str:
    from .file_utils import hash_url_to_filename
    abs_path = os.path.abspath(path)
    return os.path.join(self.extract_dir, hash_url_to_filename(abs_path))