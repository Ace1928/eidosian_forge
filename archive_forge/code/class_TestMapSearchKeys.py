from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
class TestMapSearchKeys(TestCaseWithStore):

    def test_default_chk_map_uses_flat_search_key(self):
        chkmap = chk_map.CHKMap(self.get_chk_bytes(), None)
        self.assertEqual(b'1', chkmap._search_key_func((b'1',)))
        self.assertEqual(b'1\x002', chkmap._search_key_func((b'1', b'2')))
        self.assertEqual(b'1\x002\x003', chkmap._search_key_func((b'1', b'2', b'3')))

    def test_search_key_is_passed_to_root_node(self):
        chkmap = chk_map.CHKMap(self.get_chk_bytes(), None, search_key_func=_test_search_key)
        self.assertIs(_test_search_key, chkmap._search_key_func)
        self.assertEqual(b'test:1\x002\x003', chkmap._search_key_func((b'1', b'2', b'3')))
        self.assertEqual(b'test:1\x002\x003', chkmap._root_node._search_key((b'1', b'2', b'3')))

    def test_search_key_passed_via__ensure_root(self):
        chk_bytes = self.get_chk_bytes()
        chkmap = chk_map.CHKMap(chk_bytes, None, search_key_func=_test_search_key)
        root_key = chkmap._save()
        chkmap = chk_map.CHKMap(chk_bytes, root_key, search_key_func=_test_search_key)
        chkmap._ensure_root()
        self.assertEqual(b'test:1\x002\x003', chkmap._root_node._search_key((b'1', b'2', b'3')))

    def test_search_key_with_internal_node(self):
        chk_bytes = self.get_chk_bytes()
        chkmap = chk_map.CHKMap(chk_bytes, None, search_key_func=_test_search_key)
        chkmap._root_node.set_maximum_size(10)
        chkmap.map((b'1',), b'foo')
        chkmap.map((b'2',), b'bar')
        chkmap.map((b'3',), b'baz')
        self.assertEqualDiff("'' InternalNode\n  'test:1' LeafNode\n      ('1',) 'foo'\n  'test:2' LeafNode\n      ('2',) 'bar'\n  'test:3' LeafNode\n      ('3',) 'baz'\n", chkmap._dump_tree())
        root_key = chkmap._save()
        chkmap = chk_map.CHKMap(chk_bytes, root_key, search_key_func=_test_search_key)
        self.assertEqualDiff("'' InternalNode\n  'test:1' LeafNode\n      ('1',) 'foo'\n  'test:2' LeafNode\n      ('2',) 'bar'\n  'test:3' LeafNode\n      ('3',) 'baz'\n", chkmap._dump_tree())

    def test_search_key_16(self):
        chk_bytes = self.get_chk_bytes()
        chkmap = chk_map.CHKMap(chk_bytes, None, search_key_func=chk_map._search_key_16)
        chkmap._root_node.set_maximum_size(10)
        chkmap.map((b'1',), b'foo')
        chkmap.map((b'2',), b'bar')
        chkmap.map((b'3',), b'baz')
        self.assertEqualDiff("'' InternalNode\n  '1' LeafNode\n      ('2',) 'bar'\n  '6' LeafNode\n      ('3',) 'baz'\n  '8' LeafNode\n      ('1',) 'foo'\n", chkmap._dump_tree())
        root_key = chkmap._save()
        chkmap = chk_map.CHKMap(chk_bytes, root_key, search_key_func=chk_map._search_key_16)
        self.assertEqual([((b'1',), b'foo')], list(chkmap.iteritems([(b'1',)])))
        self.assertEqualDiff("'' InternalNode\n  '1' LeafNode\n      ('2',) 'bar'\n  '6' LeafNode\n      ('3',) 'baz'\n  '8' LeafNode\n      ('1',) 'foo'\n", chkmap._dump_tree())

    def test_search_key_255(self):
        chk_bytes = self.get_chk_bytes()
        chkmap = chk_map.CHKMap(chk_bytes, None, search_key_func=chk_map._search_key_255)
        chkmap._root_node.set_maximum_size(10)
        chkmap.map((b'1',), b'foo')
        chkmap.map((b'2',), b'bar')
        chkmap.map((b'3',), b'baz')
        self.assertEqualDiff("'' InternalNode\n  '\\x1a' LeafNode\n      ('2',) 'bar'\n  'm' LeafNode\n      ('3',) 'baz'\n  '\\x83' LeafNode\n      ('1',) 'foo'\n", chkmap._dump_tree(encoding='latin1'))
        root_key = chkmap._save()
        chkmap = chk_map.CHKMap(chk_bytes, root_key, search_key_func=chk_map._search_key_255)
        self.assertEqual([((b'1',), b'foo')], list(chkmap.iteritems([(b'1',)])))
        self.assertEqualDiff("'' InternalNode\n  '\\x1a' LeafNode\n      ('2',) 'bar'\n  'm' LeafNode\n      ('3',) 'baz'\n  '\\x83' LeafNode\n      ('1',) 'foo'\n", chkmap._dump_tree(encoding='latin1'))

    def test_search_key_collisions(self):
        chkmap = chk_map.CHKMap(self.get_chk_bytes(), None, search_key_func=_search_key_single)
        chkmap._root_node.set_maximum_size(20)
        chkmap.map((b'1',), b'foo')
        chkmap.map((b'2',), b'bar')
        chkmap.map((b'3',), b'baz')
        self.assertEqualDiff("'' LeafNode\n      ('1',) 'foo'\n      ('2',) 'bar'\n      ('3',) 'baz'\n", chkmap._dump_tree())