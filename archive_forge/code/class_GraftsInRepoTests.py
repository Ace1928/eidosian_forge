import os
import shutil
import tempfile
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import Tree
from ..repo import MemoryRepo, Repo, parse_graftpoints, serialize_graftpoints
class GraftsInRepoTests(GraftsInRepositoryBase, TestCase):

    def setUp(self):
        super().setUp()
        self._repo_dir = os.path.join(tempfile.mkdtemp())
        r = self._repo = Repo.init(self._repo_dir)
        self.addCleanup(shutil.rmtree, self._repo_dir)
        self._shas = []
        commit_kwargs = {'committer': b'Test Committer <test@nodomain.com>', 'author': b'Test Author <test@nodomain.com>', 'commit_timestamp': 12395, 'commit_timezone': 0, 'author_timestamp': 12395, 'author_timezone': 0}
        self._shas.append(r.do_commit(b'empty commit', **commit_kwargs))
        self._shas.append(r.do_commit(b'empty commit', **commit_kwargs))
        self._shas.append(r.do_commit(b'empty commit', **commit_kwargs))

    def test_init_with_empty_info_grafts(self):
        r = self._repo
        r._put_named_file(os.path.join('info', 'grafts'), b'')
        r = Repo(self._repo_dir)
        self.assertEqual({}, r._graftpoints)

    def test_init_with_info_grafts(self):
        r = self._repo
        r._put_named_file(os.path.join('info', 'grafts'), self._shas[-1] + b' ' + self._shas[0])
        r = Repo(self._repo_dir)
        self.assertEqual({self._shas[-1]: [self._shas[0]]}, r._graftpoints)