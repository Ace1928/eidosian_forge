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
def _do_extract(self, output_path: str, force_extract: bool) -> bool:
    return force_extract or (not os.path.isfile(output_path) and (not (os.path.isdir(output_path) and os.listdir(output_path))))