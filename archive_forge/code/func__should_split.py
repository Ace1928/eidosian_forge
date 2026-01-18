import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _should_split(self, change):
    if self._rewrite_threshold is None or change.type != CHANGE_MODIFY or change.old.sha == change.new.sha:
        return False
    old_obj = self._store[change.old.sha]
    new_obj = self._store[change.new.sha]
    return _similarity_score(old_obj, new_obj) < self._rewrite_threshold