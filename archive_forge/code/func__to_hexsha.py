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
def _to_hexsha(self, sha):
    if len(sha) == 40:
        return sha
    elif len(sha) == 20:
        return sha_to_hex(sha)
    else:
        raise ValueError(f'Invalid sha {sha!r}')