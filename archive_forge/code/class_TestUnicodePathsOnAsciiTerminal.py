from breezy import conflicts, tests, workingtree
from breezy.tests import features, script
class TestUnicodePathsOnAsciiTerminal(TestUnicodePaths):
    """Undisplayable unicode characters in conflicts should be escaped"""
    encoding = 'ascii'

    def setUp(self):
        self.skipTest('Need to decide if replacing is the desired behaviour')

    def _as_output(self, text):
        return text.encode(self.encoding, 'replace')