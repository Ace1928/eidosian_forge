from breezy import tests
from breezy.memorytree import MemoryTree
from breezy.tests import TestCaseWithTransport
from breezy.treebuilder import AlreadyBuilding, NotBuilding, TreeBuilder
class FakeTree:
    """A pretend tree to test the calls made by TreeBuilder."""

    def __init__(self):
        self._calls = []

    def lock_tree_write(self):
        self._calls.append('lock_tree_write')

    def unlock(self):
        self._calls.append('unlock')