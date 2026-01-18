import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
def _add_cached_pack(self, base_name, pack):
    """Add a newly appeared pack to the cache by path."""
    prev_pack = self._pack_cache.get(base_name)
    if prev_pack is not pack:
        self._pack_cache[base_name] = pack
        if prev_pack:
            prev_pack.close()