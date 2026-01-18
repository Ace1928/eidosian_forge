import os
import shutil
import tempfile
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import Tree
from ..repo import MemoryRepo, Repo, parse_graftpoints, serialize_graftpoints
class GraftsInMemoryRepoTests(GraftsInRepositoryBase, TestCase):

    def setUp(self):
        super().setUp()
        r = self._repo = MemoryRepo()
        self._shas = []
        tree = Tree()
        commit_kwargs = {'committer': b'Test Committer <test@nodomain.com>', 'author': b'Test Author <test@nodomain.com>', 'commit_timestamp': 12395, 'commit_timezone': 0, 'author_timestamp': 12395, 'author_timezone': 0, 'tree': tree.id}
        self._shas.append(r.do_commit(b'empty commit', **commit_kwargs))
        self._shas.append(r.do_commit(b'empty commit', **commit_kwargs))
        self._shas.append(r.do_commit(b'empty commit', **commit_kwargs))