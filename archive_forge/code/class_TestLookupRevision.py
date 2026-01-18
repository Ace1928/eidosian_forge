from breezy import tests
class TestLookupRevision(tests.TestCaseWithTransport):

    def test_lookup_revison_directory(self):
        """Test --directory option"""
        tree = self.make_branch_and_tree('a')
        tree.commit('This revision', rev_id=b'abcd')
        out, err = self.run_bzr(['lookup-revision', '-d', 'a', '1'])
        self.assertEqual('abcd\n', out)
        self.assertEqual('', err)