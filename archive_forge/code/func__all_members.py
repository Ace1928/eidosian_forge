import abc
import os
import bz2
import gzip
import lzma
import shutil
from zipfile import ZipFile
from tarfile import TarFile
from .utils import get_logger
def _all_members(self, fname):
    """Return all members from a given archive."""
    with TarFile.open(fname, 'r') as tar_file:
        return [info.name for info in tar_file.getmembers()]