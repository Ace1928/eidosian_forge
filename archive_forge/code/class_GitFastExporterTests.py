import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
class GitFastExporterTests(TestCase):
    """Tests for the GitFastExporter tests."""

    def setUp(self):
        super().setUp()
        self.store = MemoryObjectStore()
        self.stream = BytesIO()
        try:
            from ..fastexport import GitFastExporter
        except ImportError as exc:
            raise SkipTest('python-fastimport not available') from exc
        self.fastexporter = GitFastExporter(self.stream, self.store)

    def test_emit_blob(self):
        b = Blob()
        b.data = b'fooBAR'
        self.fastexporter.emit_blob(b)
        self.assertEqual(b'blob\nmark :1\ndata 6\nfooBAR\n', self.stream.getvalue())

    def test_emit_commit(self):
        b = Blob()
        b.data = b'FOO'
        t = Tree()
        t.add(b'foo', stat.S_IFREG | 420, b.id)
        c = Commit()
        c.committer = c.author = b'Jelmer <jelmer@host>'
        c.author_time = c.commit_time = 1271345553
        c.author_timezone = c.commit_timezone = 0
        c.message = b'msg'
        c.tree = t.id
        self.store.add_objects([(b, None), (t, None), (c, None)])
        self.fastexporter.emit_commit(c, b'refs/heads/master')
        self.assertEqual(b'blob\nmark :1\ndata 3\nFOO\ncommit refs/heads/master\nmark :2\nauthor Jelmer <jelmer@host> 1271345553 +0000\ncommitter Jelmer <jelmer@host> 1271345553 +0000\ndata 3\nmsg\nM 644 :1 foo\n', self.stream.getvalue())