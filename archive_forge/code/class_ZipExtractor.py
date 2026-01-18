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
class ZipExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08']

    @classmethod
    def is_extractable(cls, path: Union[Path, str], magic_number: bytes=b'') -> bool:
        if super().is_extractable(path, magic_number=magic_number):
            return True
        try:
            from zipfile import _CD_SIGNATURE, _ECD_DISK_NUMBER, _ECD_DISK_START, _ECD_ENTRIES_TOTAL, _ECD_OFFSET, _ECD_SIZE, _EndRecData, sizeCentralDir, stringCentralDir, structCentralDir
            with open(path, 'rb') as fp:
                endrec = _EndRecData(fp)
                if endrec:
                    if endrec[_ECD_ENTRIES_TOTAL] == 0 and endrec[_ECD_SIZE] == 0 and (endrec[_ECD_OFFSET] == 0):
                        return True
                    elif endrec[_ECD_DISK_NUMBER] == endrec[_ECD_DISK_START]:
                        fp.seek(endrec[_ECD_OFFSET])
                        if fp.tell() == endrec[_ECD_OFFSET] and endrec[_ECD_SIZE] >= sizeCentralDir:
                            data = fp.read(sizeCentralDir)
                            if len(data) == sizeCentralDir:
                                centdir = struct.unpack(structCentralDir, data)
                                if centdir[_CD_SIGNATURE] == stringCentralDir:
                                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        os.makedirs(output_path, exist_ok=True)
        with zipfile.ZipFile(input_path, 'r') as zip_file:
            zip_file.extractall(output_path)
            zip_file.close()