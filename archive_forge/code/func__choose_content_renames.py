import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _choose_content_renames(self):
    self._candidates.sort()
    delete_paths = set()
    add_paths = set()
    for _, change in self._candidates:
        new_path = change.new.path
        if new_path in add_paths:
            continue
        old_path = change.old.path
        orig_type = change.type
        if old_path in delete_paths:
            change = TreeChange(CHANGE_COPY, change.old, change.new)
        if orig_type != CHANGE_COPY:
            delete_paths.add(old_path)
        add_paths.add(new_path)
        self._changes.append(change)
    self._prune(add_paths, delete_paths)