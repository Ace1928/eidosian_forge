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
def _do_test_parse_tree(self, parse_tree):
    dir = os.path.join(os.path.dirname(__file__), '..', '..', 'testdata', 'trees')
    o = Tree.from_path(hex_to_filename(dir, tree_sha))
    self.assertEqual([(b'a', 33188, a_sha), (b'b', 33188, b_sha)], list(parse_tree(o.as_raw_string())))
    broken_tree = b'0100644 foo\x00' + hex_to_sha(a_sha)

    def eval_parse_tree(*args, **kwargs):
        return list(parse_tree(*args, **kwargs))
    self.assertEqual([(b'foo', 33188, a_sha)], eval_parse_tree(broken_tree))
    self.assertRaises(ObjectFormatException, eval_parse_tree, broken_tree, strict=True)