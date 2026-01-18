from io import StringIO
import testtools
from fastimport import commands, parser
from fastimport.reftracker import RefTracker
from :100
from :101
from :100
from :100
from :100
from :100
from :102
from :100
from :100
from :100
class TestHeadTracking(testtools.TestCase):

    def assertHeads(self, input, expected):
        s = StringIO(input)
        p = parser.ImportParser(s)
        reftracker = RefTracker()
        for cmd in p.iter_commands():
            if isinstance(cmd, commands.CommitCommand):
                reftracker.track_heads(cmd)
                list(cmd.iter_files())
            elif isinstance(cmd, commands.ResetCommand):
                if cmd.from_ is not None:
                    reftracker.track_heads_for_ref(cmd.ref, cmd.from_)
        self.assertEqual(reftracker.heads, expected)

    def test_mainline(self):
        self.assertHeads(_SAMPLE_MAINLINE, {':102': {'refs/heads/master'}})

    def test_two_heads(self):
        self.assertHeads(_SAMPLE_TWO_HEADS, {':101': {'refs/heads/mybranch'}, ':102': {'refs/heads/master'}})

    def test_two_branches_merged(self):
        self.assertHeads(_SAMPLE_TWO_BRANCHES_MERGED, {':103': {'refs/heads/master'}})

    def test_reset(self):
        self.assertHeads(_SAMPLE_RESET, {':100': {'refs/heads/master', 'refs/remotes/origin/master'}})

    def test_reset_with_more_commits(self):
        self.assertHeads(_SAMPLE_RESET_WITH_MORE_COMMITS, {':101': {'refs/remotes/origin/master'}})