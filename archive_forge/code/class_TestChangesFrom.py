import socket
from breezy import revision
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestChangesFrom(TestCaseWithWorkingTree):

    def setUp(self):
        super().setUp()
        self.tree = self.make_branch_and_tree('tree')
        files = ['a', 'b/', 'b/c']
        self.build_tree(files, transport=self.tree.controldir.root_transport)
        self.tree.add(files)
        self.tree.commit('initial tree')

    def test_unknown(self):
        self.build_tree(['tree/unknown'])
        d = self.tree.changes_from(self.tree.basis_tree())
        self.assertEqual([], d.added)
        self.assertEqual([], d.removed)
        self.assertEqual([], d.renamed)
        self.assertEqual([], d.copied)
        self.assertEqual([], d.modified)

    def test_unknown_specific_file(self):
        self.build_tree(['tree/unknown'])
        empty_tree = self.tree.branch.repository.revision_tree(revision.NULL_REVISION)
        d = self.tree.changes_from(empty_tree, specific_files=['unknown'])
        self.assertEqual([], d.added)
        self.assertEqual([], d.removed)
        self.assertEqual([], d.renamed)
        self.assertEqual([], d.copied)
        self.assertEqual([], d.modified)

    def test_socket(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind('tree/socketpath')
        s.listen(1)
        empty_tree = self.tree.branch.repository.revision_tree(revision.NULL_REVISION)
        d = self.tree.changes_from(empty_tree, specific_files=['socketpath'], want_unversioned=True)
        self.assertEqual([], d.added)
        self.assertEqual([], d.removed)
        self.assertEqual([], d.renamed)
        self.assertEqual([], d.copied)
        self.assertEqual([], d.modified)
        self.assertIn([x.path[1] for x in d.unversioned], [['socketpath'], []])