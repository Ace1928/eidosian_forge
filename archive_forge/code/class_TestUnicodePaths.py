from breezy import conflicts, tests, workingtree
from breezy.tests import features, script
class TestUnicodePaths(tests.TestCaseWithTransport):
    """Unicode characters in conflicts should be displayed properly"""
    _test_needs_features = [features.UnicodeFilenameFeature]
    encoding = 'UTF-8'

    def _as_output(self, text):
        return text

    def test_messages(self):
        """Conflict messages involving non-ascii paths are displayed okay"""
        make_tree_with_conflicts(self, 'branch', prefix='§')
        out, err = self.run_bzr(['conflicts', '-d', 'branch'], encoding=self.encoding)
        self.assertEqual(out, 'Text conflict in §_other_file\nPath conflict: §dir3 / §dir2\nText conflict in §file\n')
        self.assertEqual(err, '')

    def test_text_conflict_paths(self):
        """Text conflicts on non-ascii paths are displayed okay"""
        make_tree_with_conflicts(self, 'branch', prefix='§')
        out, err = self.run_bzr(['conflicts', '-d', 'branch', '--text'], encoding=self.encoding)
        self.assertEqual(out, '§_other_file\n§file\n')
        self.assertEqual(err, '')