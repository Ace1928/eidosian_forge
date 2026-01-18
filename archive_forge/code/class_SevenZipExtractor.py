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
class SevenZipExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b"7z\xbc\xaf'\x1c"]

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        if not config.PY7ZR_AVAILABLE:
            raise ImportError('Please pip install py7zr')
        import py7zr
        os.makedirs(output_path, exist_ok=True)
        with py7zr.SevenZipFile(input_path, 'r') as archive:
            archive.extractall(output_path)