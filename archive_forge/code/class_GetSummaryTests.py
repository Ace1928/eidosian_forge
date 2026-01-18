from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
class GetSummaryTests(TestCase):

    def test_simple(self):
        c = Commit()
        c.committer = c.author = b'Jelmer <jelmer@samba.org>'
        c.commit_time = c.author_time = 1271350201
        c.commit_timezone = c.author_timezone = 0
        c.message = b'This is the first line\nAnd this is the second line.\n'
        c.tree = Tree().id
        self.assertEqual('This-is-the-first-line', get_summary(c))