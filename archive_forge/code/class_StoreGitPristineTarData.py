import stat
from base64 import standard_b64encode
from dulwich.objects import Blob, Tree
from dulwich.repo import MemoryRepo as GitMemoryRepo
from ...revision import Revision
from ...tests import TestCase
from ..pristine_tar import (get_pristine_tar_tree, read_git_pristine_tar_data,
class StoreGitPristineTarData(TestCase):

    def test_store_new(self):
        r = GitMemoryRepo()
        cid = store_git_pristine_tar_data(r, b'foo', b'mydelta', b'myid')
        tree = get_pristine_tar_tree(r)
        self.assertEqual((stat.S_IFREG | 420, b'7b02de8ac4162e64f402c43487d8a40a505482e1'), tree[b'README'])
        self.assertEqual(r[cid].tree, tree.id)
        self.assertEqual(r[tree[b'foo.delta'][1]].data, b'mydelta')
        self.assertEqual(r[tree[b'foo.id'][1]].data, b'myid')
        self.assertEqual((b'mydelta', b'myid'), read_git_pristine_tar_data(r, b'foo'))