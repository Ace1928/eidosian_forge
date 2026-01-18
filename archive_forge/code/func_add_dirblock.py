import os
from breezy.tests.features import SymlinkFeature
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def add_dirblock(path, kind):
    dirblock = DirBlock(tree, path)
    if file_status != self.unknown:
        dirblock.inventory_kind = kind
    if file_status != self.missing:
        dirblock.disk_kind = kind
        dirblock.stat = os.lstat(dirblock.relpath)
    dirblocks.append(dirblock)