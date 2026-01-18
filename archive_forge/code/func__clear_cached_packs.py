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
def _clear_cached_packs(self):
    pack_cache = self._pack_cache
    self._pack_cache = {}
    while pack_cache:
        name, pack = pack_cache.popitem()
        pack.close()