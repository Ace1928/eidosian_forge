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
class TarExtractor(BaseExtractor):

    @classmethod
    def is_extractable(cls, path: Union[Path, str], **kwargs) -> bool:
        return tarfile.is_tarfile(path)

    @staticmethod
    def safemembers(members, output_path):
        """
        Fix for CVE-2007-4559
        Desc:
            Directory traversal vulnerability in the (1) extract and (2) extractall functions in the tarfile
            module in Python allows user-assisted remote attackers to overwrite arbitrary files via a .. (dot dot)
            sequence in filenames in a TAR archive, a related issue to CVE-2001-1267.
        See: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2007-4559
        From: https://stackoverflow.com/a/10077309
        """

        def resolved(path: str) -> str:
            return os.path.realpath(os.path.abspath(path))

        def badpath(path: str, base: str) -> bool:
            return not resolved(os.path.join(base, path)).startswith(base)

        def badlink(info, base: str) -> bool:
            tip = resolved(os.path.join(base, os.path.dirname(info.name)))
            return badpath(info.linkname, base=tip)
        base = resolved(output_path)
        for finfo in members:
            if badpath(finfo.name, base):
                logger.error(f'Extraction of {finfo.name} is blocked (illegal path)')
            elif finfo.issym() and badlink(finfo, base):
                logger.error(f'Extraction of {finfo.name} is blocked: Symlink to {finfo.linkname}')
            elif finfo.islnk() and badlink(finfo, base):
                logger.error(f'Extraction of {finfo.name} is blocked: Hard link to {finfo.linkname}')
            else:
                yield finfo

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        os.makedirs(output_path, exist_ok=True)
        tar_file = tarfile.open(input_path)
        tar_file.extractall(output_path, members=TarExtractor.safemembers(tar_file, output_path))
        tar_file.close()