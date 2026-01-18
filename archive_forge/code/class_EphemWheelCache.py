import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pip._vendor.packaging.tags import Tag, interpreter_name, interpreter_version
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import InvalidWheelFilename
from pip._internal.models.direct_url import DirectUrl
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
from pip._internal.utils.urls import path_to_url
class EphemWheelCache(SimpleWheelCache):
    """A SimpleWheelCache that creates it's own temporary cache directory"""

    def __init__(self) -> None:
        self._temp_dir = TempDirectory(kind=tempdir_kinds.EPHEM_WHEEL_CACHE, globally_managed=True)
        super().__init__(self._temp_dir.path)