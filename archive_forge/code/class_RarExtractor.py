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
class RarExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b'Rar!\x1a\x07\x00', b'Rar!\x1a\x07\x01\x00']

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        if not config.RARFILE_AVAILABLE:
            raise ImportError('Please pip install rarfile')
        import rarfile
        os.makedirs(output_path, exist_ok=True)
        rf = rarfile.RarFile(input_path)
        rf.extractall(output_path)
        rf.close()