from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
class MatchesTreeChanges(Matcher):
    """A matcher that checks that tree changes match expected contents."""

    def __init__(self, old_tree, new_tree, expected):
        Matcher.__init__(self)
        expected = [TreeChange(*x) if isinstance(x, tuple) else x for x in expected]
        self.use_inventory_tree_changes = old_tree.supports_file_ids and new_tree.supports_file_ids
        self.expected = expected
        self.old_tree = old_tree
        self.new_tree = new_tree

    @staticmethod
    def _convert_to_inventory_tree_changes(old_tree, new_tree, expected):
        from ..bzr.inventorytree import InventoryTreeChange
        rich_expected = []

        def get_parent_id(t, p):
            if p:
                return t.path2id(osutils.dirname(p))
            else:
                return None
        for c in expected:
            if c.path[0] is not None:
                file_id = old_tree.path2id(c.path[0])
            else:
                file_id = new_tree.path2id(c.path[1])
            old_parent_id = get_parent_id(old_tree, c.path[0])
            new_parent_id = get_parent_id(new_tree, c.path[1])
            rich_expected.append(InventoryTreeChange(file_id=file_id, parent_id=(old_parent_id, new_parent_id), path=c.path, changed_content=c.changed_content, versioned=c.versioned, name=c.name, kind=c.kind, executable=c.executable, copied=c.copied))
        return rich_expected

    def __str__(self):
        return '<MatchesTreeChanges(%r)>' % self.expected

    def match(self, actual):
        from ..bzr.inventorytree import InventoryTreeChange
        actual = list(actual)
        if self.use_inventory_tree_changes or (actual and isinstance(actual[0], InventoryTreeChange)):
            expected = self._convert_to_inventory_tree_changes(self.old_tree, self.new_tree, self.expected)
        else:
            expected = self.expected
        if self.use_inventory_tree_changes:
            actual = self._convert_to_inventory_tree_changes(self.old_tree, self.new_tree, actual)
        return Equals(expected).match(actual)