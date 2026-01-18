import os
from typing import Set
from pyinotify import (IN_ATTRIB, IN_CLOSE_WRITE, IN_CREATE, IN_DELETE,
from .workingtree import WorkingTree
@property
def _created(self):
    return self._process.created