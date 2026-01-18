from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def _do_test_count_blocks_no_newline(self, count_blocks):
    blob = make_object(Blob, data=b'a\na')
    self.assertBlockCountEqual({b'a\n': 2, b'a': 1}, _count_blocks(blob))