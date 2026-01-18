import email.message
import email.parser
import logging
import os
import zipfile
from typing import Collection, Iterable, Iterator, List, Mapping, NamedTuple, Optional
from pip._vendor import pkg_resources
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.exceptions import InvalidWheel, NoneMetadataError, UnsupportedWheel
from pip._internal.utils.egg_link import egg_link_path_from_location
from pip._internal.utils.misc import display_path, normalize_path
from pip._internal.utils.wheel import parse_wheel, read_wheel_metadata_file
from .base import (
class InMemoryMetadata:
    """IMetadataProvider that reads metadata files from a dictionary.

    This also maps metadata decoding exceptions to our internal exception type.
    """

    def __init__(self, metadata: Mapping[str, bytes], wheel_name: str) -> None:
        self._metadata = metadata
        self._wheel_name = wheel_name

    def has_metadata(self, name: str) -> bool:
        return name in self._metadata

    def get_metadata(self, name: str) -> str:
        try:
            return self._metadata[name].decode()
        except UnicodeDecodeError as e:
            raise UnsupportedWheel(f'Error decoding metadata for {self._wheel_name}: {e} in {name} file')

    def get_metadata_lines(self, name: str) -> Iterable[str]:
        return pkg_resources.yield_lines(self.get_metadata(name))

    def metadata_isdir(self, name: str) -> bool:
        return False

    def metadata_listdir(self, name: str) -> List[str]:
        return []

    def run_script(self, script_name: str, namespace: str) -> None:
        pass