import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _join_modifies(self):
    if self._rewrite_threshold is None:
        return
    modifies = {}
    delete_map = {d.old.path: d for d in self._deletes}
    for add in self._adds:
        path = add.new.path
        delete = delete_map.get(path)
        if delete is not None and stat.S_IFMT(delete.old.mode) == stat.S_IFMT(add.new.mode):
            modifies[path] = TreeChange(CHANGE_MODIFY, delete.old, add.new)
    self._adds = [a for a in self._adds if a.new.path not in modifies]
    self._deletes = [a for a in self._deletes if a.new.path not in modifies]
    self._changes += modifies.values()