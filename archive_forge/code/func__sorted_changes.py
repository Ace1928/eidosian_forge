import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _sorted_changes(self):
    result = []
    result.extend(self._adds)
    result.extend(self._deletes)
    result.extend(self._changes)
    result.sort(key=_tree_change_key)
    return result