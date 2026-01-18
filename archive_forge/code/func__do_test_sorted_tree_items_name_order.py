import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
def _do_test_sorted_tree_items_name_order(self, sorted_tree_items):
    self.assertEqual([TreeEntry(b'a', stat.S_IFDIR, b'd80c186a03f423a81b39df39dc87fd269736ca86'), TreeEntry(b'a.c', 33261, b'd80c186a03f423a81b39df39dc87fd269736ca86'), TreeEntry(b'a/c', stat.S_IFDIR, b'd80c186a03f423a81b39df39dc87fd269736ca86')], list(sorted_tree_items(_TREE_ITEMS, True)))