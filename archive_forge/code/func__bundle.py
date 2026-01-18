import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
def _bundle(self):
    if self.patch_type == 'bundle':
        return self.patch
    else:
        return None