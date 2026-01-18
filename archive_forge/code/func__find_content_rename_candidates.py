import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _find_content_rename_candidates(self):
    candidates = self._candidates = []
    if not self._should_find_content_renames():
        return
    block_cache = {}
    check_paths = self._rename_threshold is not None
    for delete in self._deletes:
        if S_ISGITLINK(delete.old.mode):
            continue
        old_sha = delete.old.sha
        old_obj = self._store[old_sha]
        block_cache[old_sha] = _count_blocks(old_obj)
        for add in self._adds:
            if stat.S_IFMT(delete.old.mode) != stat.S_IFMT(add.new.mode):
                continue
            new_obj = self._store[add.new.sha]
            score = _similarity_score(old_obj, new_obj, block_cache=block_cache)
            if score > self._rename_threshold:
                new_type = self._rename_type(check_paths, delete, add)
                rename = TreeChange(new_type, delete.old, add.new)
                candidates.append((-score, rename))