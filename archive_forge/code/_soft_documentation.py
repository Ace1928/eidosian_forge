from __future__ import annotations
import os
import sys
from contextlib import suppress
from errno import EACCES, EEXIST
from pathlib import Path
from ._api import BaseFileLock
from ._util import ensure_directory_exists, raise_on_not_writable_file
Simply watches the existence of the lock file.