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
class WorkspaceDirty(BzrError):
    _fmt = 'The directory %(path)s has pending changes.'

    def __init__(self, tree, subpath):
        self.tree = tree
        self.subpath = subpath
        BzrError.__init__(self, path=tree.abspath(subpath))