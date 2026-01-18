import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _prune_unchanged(self):
    if self._want_unchanged:
        return
    self._deletes = [d for d in self._deletes if d.type != CHANGE_UNCHANGED]