from __future__ import absolute_import
import os
from .. import Utils
def _bad_access(self):
    raise RuntimeError(repr(self))