import email.message
import importlib.metadata
import os
import pathlib
import zipfile
from typing import (
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.exceptions import InvalidWheel, UnsupportedWheel
from pip._internal.metadata.base import (
from pip._internal.utils.misc import normalize_path
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.wheel import parse_wheel, read_wheel_metadata_file
from ._compat import BasePath, get_dist_name
class WheelDistribution(importlib.metadata.Distribution):
    """An ``importlib.metadata.Distribution`` read from a wheel.

    Although ``importlib.metadata.PathDistribution`` accepts ``zipfile.Path``,
    its implementation is too "lazy" for pip's needs (we can't keep the ZipFile
    handle open for the entire lifetime of the distribution object).

    This implementation eagerly reads the entire metadata directory into the
    memory instead, and operates from that.
    """

    def __init__(self, files: Mapping[pathlib.PurePosixPath, bytes], info_location: pathlib.PurePosixPath) -> None:
        self._files = files
        self.info_location = info_location

    @classmethod
    def from_zipfile(cls, zf: zipfile.ZipFile, name: str, location: str) -> 'WheelDistribution':
        info_dir, _ = parse_wheel(zf, name)
        paths = ((name, pathlib.PurePosixPath(name.split('/', 1)[-1])) for name in zf.namelist() if name.startswith(f'{info_dir}/'))
        files = {relpath: read_wheel_metadata_file(zf, fullpath) for fullpath, relpath in paths}
        info_location = pathlib.PurePosixPath(location, info_dir)
        return cls(files, info_location)

    def iterdir(self, path: InfoPath) -> Iterator[pathlib.PurePosixPath]:
        if pathlib.PurePosixPath(str(path)) in self._files:
            return iter(self._files)
        raise FileNotFoundError(path)

    def read_text(self, filename: str) -> Optional[str]:
        try:
            data = self._files[pathlib.PurePosixPath(filename)]
        except KeyError:
            return None
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError as e:
            wheel = self.info_location.parent
            error = f'Error decoding metadata for {wheel}: {e} in {filename} file'
            raise UnsupportedWheel(error)
        return text