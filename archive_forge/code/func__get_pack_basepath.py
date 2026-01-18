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
def _get_pack_basepath(self, entries):
    suffix = iter_sha1((entry[0] for entry in entries))
    suffix = suffix.decode('ascii')
    return os.path.join(self.pack_dir, 'pack-' + suffix)