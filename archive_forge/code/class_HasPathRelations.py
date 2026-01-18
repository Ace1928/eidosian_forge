from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
class HasPathRelations(Matcher):
    """Matcher verifies that paths have a relation to those in another tree.

    :ivar previous_tree: tree to compare to
    :ivar previous_entries: List of expected entries, as (path, previous_path) pairs.
    """

    def __init__(self, previous_tree, previous_entries):
        Matcher.__init__(self)
        self.previous_tree = previous_tree
        self.previous_entries = previous_entries

    def get_path_map(self, tree):
        """Get the (path, previous_path) pairs for the current tree."""
        previous_intertree = InterTree.get(self.previous_tree, tree)
        with tree.lock_read(), self.previous_tree.lock_read():
            for path, ie in tree.iter_entries_by_dir():
                if tree.supports_rename_tracking():
                    previous_path = previous_intertree.find_source_path(path)
                elif self.previous_tree.is_versioned(path):
                    previous_path = path
                else:
                    previous_path = None
                if previous_path:
                    kind = self.previous_tree.kind(previous_path)
                    if kind == 'directory':
                        previous_path += '/'
                if path == '':
                    yield ('', previous_path)
                else:
                    yield (path + ie.kind_character(), previous_path)

    @staticmethod
    def _strip_unreferenced_directories(entries):
        """Strip all directories that don't (in)directly contain any files.

        :param entries: List of path strings or (path, previous_path) tuples to process
        """
        directory_used = set()
        directories = []
        for path, previous_path in entries:
            if not path or path[-1] == '/':
                directories.append((path, previous_path))
            else:
                for direntry in directories:
                    if osutils.is_inside(direntry[0], path):
                        directory_used.add(direntry[0])
        for path, previous_path in entries:
            if not path.endswith('/') or path in directory_used:
                yield (path, previous_path)

    def __str__(self):
        return 'HasPathRelations({!r}, {!r})'.format(self.previous_tree, self.previous_entries)

    def match(self, tree):
        actual = list(self.get_path_map(tree))
        if not tree.has_versioned_directories():
            entries = list(self._strip_unreferenced_directories(self.previous_entries))
        else:
            entries = self.previous_entries
        if not tree.supports_rename_tracking():
            entries = [(path, path if self.previous_tree.is_versioned(path) else None) for path, previous_path in entries]
        return Equals(entries).match(actual)