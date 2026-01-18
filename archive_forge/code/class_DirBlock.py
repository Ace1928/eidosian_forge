import os
from breezy.tests.features import SymlinkFeature
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class DirBlock:
    """Object representation of the tuples returned by dirstate."""

    def __init__(self, tree, file_path, file_name=None, inventory_kind=None, stat=None, disk_kind='unknown'):
        self.file_path = file_path
        self.abspath = tree.abspath(file_path)
        self.relpath = tree.relpath(file_path)
        if file_name is None:
            file_name = os.path.split(file_path)[-1]
            if len(file_name) == 0:
                file_name = os.path.split(file_path)[-2]
        self.file_name = file_name
        self.inventory_kind = inventory_kind
        self.stat = stat
        self.disk_kind = disk_kind

    def as_tuple(self):
        return (self.relpath, self.file_name, self.disk_kind, self.stat, self.inventory_kind)

    def as_dir_tuple(self):
        return self.relpath

    def __str__(self):
        return '\nfile_path      = {!r}\nabspath        = {!r}\nrelpath        = {!r}\nfile_name      = {!r}\ninventory_kind = {!r}\nstat           = {!r}\ndisk_kind      = {!r}'.format(self.file_path, self.abspath, self.relpath, self.file_name, self.inventory_kind, self.stat, self.disk_kind)