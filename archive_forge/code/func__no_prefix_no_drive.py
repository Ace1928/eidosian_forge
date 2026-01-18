import abc
from collections import defaultdict
import collections.abc
from contextlib import contextmanager
import os
from pathlib import (  # type: ignore
import shutil
import sys
from typing import (
from urllib.parse import urlparse
from warnings import warn
from cloudpathlib.enums import FileCacheMode
from . import anypath
from .exceptions import (
@property
def _no_prefix_no_drive(self) -> str:
    return self._str[len(self.cloud_prefix) + len(self.drive):]