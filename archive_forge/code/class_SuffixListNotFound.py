from __future__ import annotations
import logging
import pkgutil
import re
from collections.abc import Sequence
from typing import cast
import requests
from requests_file import FileAdapter  # type: ignore[import-untyped]
from .cache import DiskCache
class SuffixListNotFound(LookupError):
    """A recoverable error while looking up a suffix list.

    Recoverable because you can specify backups, or use this library's bundled
    snapshot.
    """