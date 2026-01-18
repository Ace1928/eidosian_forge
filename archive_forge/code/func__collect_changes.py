import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _collect_changes(self, tree1_id, tree2_id):
    want_unchanged = self._find_copies_harder or self._want_unchanged
    for change in tree_changes(self._store, tree1_id, tree2_id, want_unchanged=want_unchanged, include_trees=self._include_trees):
        self._add_change(change)