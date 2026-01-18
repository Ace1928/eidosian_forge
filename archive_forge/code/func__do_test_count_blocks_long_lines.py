from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def _do_test_count_blocks_long_lines(self, count_blocks):
    a = b'a' * 64
    data = a + b'xxx\ny\n' + a + b'zzz\n'
    blob = make_object(Blob, data=data)
    self.assertBlockCountEqual({b'a' * 64: 128, b'xxx\n': 4, b'y\n': 2, b'zzz\n': 4}, _count_blocks(blob))