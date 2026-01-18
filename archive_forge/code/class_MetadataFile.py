import functools
import itertools
import logging
import os
import posixpath
import re
import urllib.parse
from dataclasses import dataclass
from typing import (
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
from pip._internal.utils.models import KeyBasedCompareMixin
from pip._internal.utils.urls import path_to_url, url_to_path
@dataclass(frozen=True)
class MetadataFile:
    """Information about a core metadata file associated with a distribution."""
    hashes: Optional[Dict[str, str]]

    def __post_init__(self) -> None:
        if self.hashes is not None:
            assert all((name in _SUPPORTED_HASHES for name in self.hashes))