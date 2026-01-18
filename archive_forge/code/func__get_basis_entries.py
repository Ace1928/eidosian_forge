from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def _get_basis_entries(tree):
    basis_tree = tree.basis_tree()
    with basis_tree.lock_read():
        return list(basis_tree.inventory.iter_entries_by_dir())