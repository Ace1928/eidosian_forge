import errno
import os
from io import StringIO
from ... import branch as _mod_branch
from ... import config, controldir, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import bzrdir
from ...bzr.conflicts import ConflictList, ContentsConflict, TextConflict
from ...bzr.inventory import Inventory
from ...bzr.workingtree import InventoryWorkingTree
from ...errors import PathsNotVersionedError, UnsupportedOperation
from ...mutabletree import MutableTree
from ...osutils import getcwd, pathjoin, supports_symlinks
from ...tree import TreeDirectory, TreeFile, TreeLink
from ...workingtree import SettingFileIdUnsupported, WorkingTree
from .. import TestNotApplicable, TestSkipped, features
from . import TestCaseWithWorkingTree
class TestFormatAttributes(TestCaseWithWorkingTree):

    def test_versioned_directories(self):
        self.assertSubset([self.workingtree_format.supports_versioned_directories], (True, False))

    def test_supports_setting_file_ids(self):
        self.assertSubset([self.workingtree_format.supports_setting_file_ids], (True, False))

    def test_supports_store_uncommitted(self):
        self.assertSubset([self.workingtree_format.supports_store_uncommitted], (True, False))