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
def _iter_alternate_objects(self):
    """Iterate over the SHAs of all the objects in alternate stores."""
    for alternate in self.alternates:
        yield from alternate