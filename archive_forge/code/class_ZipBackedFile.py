import collections
import compileall
import contextlib
import csv
import importlib
import logging
import os.path
import re
import shutil
import sys
import warnings
from base64 import urlsafe_b64encode
from email.message import Message
from itertools import chain, filterfalse, starmap
from typing import (
from zipfile import ZipFile, ZipInfo
from pip._vendor.distlib.scripts import ScriptMaker
from pip._vendor.distlib.util import get_export_entry
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import InstallationError
from pip._internal.locations import get_major_minor_version
from pip._internal.metadata import (
from pip._internal.models.direct_url import DIRECT_URL_METADATA_NAME, DirectUrl
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.filesystem import adjacent_tmp_file, replace
from pip._internal.utils.misc import captured_stdout, ensure_dir, hash_file, partition
from pip._internal.utils.unpacking import (
from pip._internal.utils.wheel import parse_wheel
class ZipBackedFile:

    def __init__(self, src_record_path: RecordPath, dest_path: str, zip_file: ZipFile) -> None:
        self.src_record_path = src_record_path
        self.dest_path = dest_path
        self._zip_file = zip_file
        self.changed = False

    def _getinfo(self) -> ZipInfo:
        return self._zip_file.getinfo(self.src_record_path)

    def save(self) -> None:
        parent_dir = os.path.dirname(self.dest_path)
        ensure_dir(parent_dir)
        if os.path.exists(self.dest_path):
            os.unlink(self.dest_path)
        zipinfo = self._getinfo()
        with self._zip_file.open(zipinfo) as f:
            with open(self.dest_path, 'wb') as dest:
                shutil.copyfileobj(f, dest)
        if zip_item_is_executable(zipinfo):
            set_extracted_file_to_default_mode_plus_executable(self.dest_path)