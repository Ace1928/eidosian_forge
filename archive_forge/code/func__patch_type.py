import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
def _patch_type(self):
    if self.bundle is not None:
        return 'bundle'
    elif self.patch is not None:
        return 'diff'
    else:
        return None