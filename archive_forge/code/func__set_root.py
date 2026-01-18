from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _set_root(self, ie):
    self.root = ie
    self._byid = {self.root.file_id: self.root}