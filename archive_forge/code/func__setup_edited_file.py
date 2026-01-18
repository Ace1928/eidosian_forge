from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def _setup_edited_file(self, relpath='.'):
    """Create a tree with a locally edited file."""
    tree = self.make_branch_and_tree(relpath)
    file_relpath = joinpath(relpath, 'file')
    self.build_tree_contents([(file_relpath, b'foo\ngam\n')])
    tree.add('file')
    tree.commit('add file', committer='test@host', rev_id=b'rev1')
    self.build_tree_contents([(file_relpath, b'foo\nbar\ngam\n')])
    return tree