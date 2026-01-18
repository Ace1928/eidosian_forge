import zipfile
import tarfile
import os
import shutil
import posixpath
import contextlib
from distutils.errors import DistutilsError
from ._path import ensure_directory
class UnrecognizedFormat(DistutilsError):
    """Couldn't recognize the archive type"""