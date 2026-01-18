import errno
import os
import shutil
from contextlib import ExitStack
from typing import List, Optional
from .clean_tree import iter_deletables
from .errors import BzrError, DependencyNotPresent
from .osutils import is_inside
from .trace import warning
from .transform import revert
from .transport import NoSuchFile
from .tree import Tree
from .workingtree import WorkingTree
def _stage(self) -> Optional[List[str]]:
    changed: Optional[List[str]]
    if self._dirty_tracker:
        relpaths = self._dirty_tracker.relpaths()
        self.tree.add([p for p in sorted(relpaths) if self.tree.has_filename(p) and (not self.tree.is_ignored(p))])
        changed = [p for p in relpaths if self.tree.is_versioned(p)]
    else:
        self.tree.smart_add([self.tree.abspath(self.subpath)])
        changed = [self.subpath] if self.subpath else None
    if self.tree.supports_setting_file_ids():
        from .rename_map import RenameMap
        basis_tree = self.tree.basis_tree()
        RenameMap.guess_renames(basis_tree, self.tree, dry_run=False)
    return changed