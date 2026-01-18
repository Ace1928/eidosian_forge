import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _prune(self, add_paths, delete_paths):
    self._adds = [a for a in self._adds if a.new.path not in add_paths]
    self._deletes = [d for d in self._deletes if d.old.path not in delete_paths]