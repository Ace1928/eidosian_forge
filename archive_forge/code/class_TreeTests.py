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
class TreeTests(ShaFileCheckTests):

    def test_add(self):
        myhexsha = b'd80c186a03f423a81b39df39dc87fd269736ca86'
        x = Tree()
        x.add(b'myname', 33261, myhexsha)
        self.assertEqual(x[b'myname'], (33261, myhexsha))
        self.assertEqual(b'100755 myname\x00' + hex_to_sha(myhexsha), x.as_raw_string())

    def test_simple(self):
        myhexsha = b'd80c186a03f423a81b39df39dc87fd269736ca86'
        x = Tree()
        x[b'myname'] = (33261, myhexsha)
        self.assertEqual(b'100755 myname\x00' + hex_to_sha(myhexsha), x.as_raw_string())
        self.assertEqual(b'100755 myname\x00' + hex_to_sha(myhexsha), bytes(x))

    def test_tree_update_id(self):
        x = Tree()
        x[b'a.c'] = (33261, b'd80c186a03f423a81b39df39dc87fd269736ca86')
        self.assertEqual(b'0c5c6bc2c081accfbc250331b19e43b904ab9cdd', x.id)
        x[b'a.b'] = (stat.S_IFDIR, b'd80c186a03f423a81b39df39dc87fd269736ca86')
        self.assertEqual(b'07bfcb5f3ada15bbebdfa3bbb8fd858a363925c8', x.id)

    def test_tree_iteritems_dir_sort(self):
        x = Tree()
        for name, item in _TREE_ITEMS.items():
            x[name] = item
        self.assertEqual(_SORTED_TREE_ITEMS, x.items())

    def test_tree_items_dir_sort(self):
        x = Tree()
        for name, item in _TREE_ITEMS.items():
            x[name] = item
        self.assertEqual(_SORTED_TREE_ITEMS, x.items())

    def _do_test_parse_tree(self, parse_tree):
        dir = os.path.join(os.path.dirname(__file__), '..', '..', 'testdata', 'trees')
        o = Tree.from_path(hex_to_filename(dir, tree_sha))
        self.assertEqual([(b'a', 33188, a_sha), (b'b', 33188, b_sha)], list(parse_tree(o.as_raw_string())))
        broken_tree = b'0100644 foo\x00' + hex_to_sha(a_sha)

        def eval_parse_tree(*args, **kwargs):
            return list(parse_tree(*args, **kwargs))
        self.assertEqual([(b'foo', 33188, a_sha)], eval_parse_tree(broken_tree))
        self.assertRaises(ObjectFormatException, eval_parse_tree, broken_tree, strict=True)
    test_parse_tree = functest_builder(_do_test_parse_tree, _parse_tree_py)
    test_parse_tree_extension = ext_functest_builder(_do_test_parse_tree, parse_tree)

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
    test_sorted_tree_items = functest_builder(_do_test_sorted_tree_items, _sorted_tree_items_py)
    test_sorted_tree_items_extension = ext_functest_builder(_do_test_sorted_tree_items, sorted_tree_items)

    def _do_test_sorted_tree_items_name_order(self, sorted_tree_items):
        self.assertEqual([TreeEntry(b'a', stat.S_IFDIR, b'd80c186a03f423a81b39df39dc87fd269736ca86'), TreeEntry(b'a.c', 33261, b'd80c186a03f423a81b39df39dc87fd269736ca86'), TreeEntry(b'a/c', stat.S_IFDIR, b'd80c186a03f423a81b39df39dc87fd269736ca86')], list(sorted_tree_items(_TREE_ITEMS, True)))
    test_sorted_tree_items_name_order = functest_builder(_do_test_sorted_tree_items_name_order, _sorted_tree_items_py)
    test_sorted_tree_items_name_order_extension = ext_functest_builder(_do_test_sorted_tree_items_name_order, sorted_tree_items)

    def test_check(self):
        t = Tree
        sha = hex_to_sha(a_sha)
        self.assertCheckSucceeds(t, b'100644 .a\x00' + sha)
        self.assertCheckFails(t, b'100644 \x00' + sha)
        self.assertCheckFails(t, b'100644 .\x00' + sha)
        self.assertCheckFails(t, b'100644 a/a\x00' + sha)
        self.assertCheckFails(t, b'100644 ..\x00' + sha)
        self.assertCheckFails(t, b'100644 .git\x00' + sha)
        self.assertCheckSucceeds(t, b'100644 a\x00' + sha)
        self.assertCheckSucceeds(t, b'100755 a\x00' + sha)
        self.assertCheckSucceeds(t, b'160000 a\x00' + sha)
        self.assertCheckFails(t, b'123456 a\x00' + sha)
        self.assertCheckFails(t, b'123abc a\x00' + sha)
        self.assertCheckFails(t, b'0100644 foo\x00' + sha)
        self.assertCheckFails(t, b'100644 a\x00' + b'x' * 5)
        self.assertCheckFails(t, b'100644 a\x00' + b'x' * 18 + b'\x00')
        self.assertCheckFails(t, b'100644 a\x00' + b'x' * 21 + b'\n100644 b\x00' + sha)
        sha2 = hex_to_sha(b_sha)
        self.assertCheckSucceeds(t, b'100644 a\x00' + sha + b'\n100644 b\x00' + sha)
        self.assertCheckSucceeds(t, b'100644 a\x00' + sha + b'\n100644 b\x00' + sha2)
        self.assertCheckFails(t, b'100644 a\x00' + sha + b'\n100755 a\x00' + sha2)
        self.assertCheckFails(t, b'100644 b\x00' + sha2 + b'\n100644 a\x00' + sha)

    def test_iter(self):
        t = Tree()
        t[b'foo'] = (33188, a_sha)
        self.assertEqual({b'foo'}, set(t))