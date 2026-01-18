from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def _create_merged_file(self):
    """Create a file with a pending merge and local edit."""
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('file', b'foo\ngam\n')])
    tree.add('file')
    tree.commit('add file', rev_id=b'rev1', committer='test@host')
    self.build_tree_contents([('file', b'foo\nbar\ngam\n')])
    tree.commit('right', rev_id=b'rev1.1.1', committer='test@host')
    tree.pull(tree.branch, True, b'rev1')
    self.build_tree_contents([('file', b'foo\nbaz\ngam\n')])
    tree.commit('left', rev_id=b'rev2', committer='test@host')
    tree.merge_from_branch(tree.branch, b'rev1.1.1')
    self.build_tree_contents([('file', b'local\nfoo\nbar\nbaz\ngam\n')])
    return tree