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
def _do_test_sorted_tree_items(self, sorted_tree_items):

    def do_sort(entries):
        return list(sorted_tree_items(entries, False))
    actual = do_sort(_TREE_ITEMS)
    self.assertEqual(_SORTED_TREE_ITEMS, actual)
    self.assertIsInstance(actual[0], TreeEntry)
    errors = (TypeError, ValueError, AttributeError)
    self.assertRaises(errors, do_sort, b'foo')
    self.assertRaises(errors, do_sort, {b'foo': (1, 2, 3)})
    myhexsha = b'd80c186a03f423a81b39df39dc87fd269736ca86'
    self.assertRaises(errors, do_sort, {b'foo': (b'xxx', myhexsha)})
    self.assertRaises(errors, do_sort, {b'foo': (33261, 12345)})