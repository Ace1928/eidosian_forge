from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
class TestIterInterestingNodes(TestCaseWithExampleMaps):

    def get_map_key(self, a_dict, maximum_size=10):
        c_map = self.get_map(a_dict, maximum_size=maximum_size)
        return c_map.key()

    def assertIterInteresting(self, records, items, interesting_keys, old_keys):
        """Check the result of iter_interesting_nodes.

        Note that we no longer care how many steps are taken, etc, just that
        the right contents are returned.

        :param records: A list of record keys that should be yielded
        :param items: A list of items (key,value) that should be yielded.
        """
        store = self.get_chk_bytes()
        store._search_key_func = chk_map._search_key_plain
        iter_nodes = chk_map.iter_interesting_nodes(store, interesting_keys, old_keys)
        record_keys = []
        all_items = []
        for record, new_items in iter_nodes:
            if record is not None:
                record_keys.append(record.key)
            if new_items:
                all_items.extend(new_items)
        self.assertEqual(sorted(records), sorted(record_keys))
        self.assertEqual(sorted(items), sorted(all_items))

    def test_empty_to_one_keys(self):
        target = self.get_map_key({(b'a',): b'content'})
        self.assertIterInteresting([target], [((b'a',), b'content')], [target], [])

    def test_none_to_one_key(self):
        basis = self.get_map_key({})
        target = self.get_map_key({(b'a',): b'content'})
        self.assertIterInteresting([target], [((b'a',), b'content')], [target], [basis])

    def test_one_to_none_key(self):
        basis = self.get_map_key({(b'a',): b'content'})
        target = self.get_map_key({})
        self.assertIterInteresting([target], [], [target], [basis])

    def test_common_pages(self):
        basis = self.get_map_key({(b'a',): b'content', (b'b',): b'content', (b'c',): b'content'})
        target = self.get_map_key({(b'a',): b'content', (b'b',): b'other content', (b'c',): b'content'})
        target_map = CHKMap(self.get_chk_bytes(), target)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('a',) 'content'\n  'b' LeafNode\n      ('b',) 'other content'\n  'c' LeafNode\n      ('c',) 'content'\n", target_map._dump_tree())
        b_key = target_map._root_node._items[b'b'].key()
        self.assertIterInteresting([target, b_key], [((b'b',), b'other content')], [target], [basis])

    def test_common_sub_page(self):
        basis = self.get_map_key({(b'aaa',): b'common', (b'c',): b'common'})
        target = self.get_map_key({(b'aaa',): b'common', (b'aab',): b'new', (b'c',): b'common'})
        target_map = CHKMap(self.get_chk_bytes(), target)
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'common'\n    'aab' LeafNode\n      ('aab',) 'new'\n  'c' LeafNode\n      ('c',) 'common'\n", target_map._dump_tree())
        a_key = target_map._root_node._items[b'a'].key()
        aab_key = target_map._root_node._items[b'a']._items[b'aab'].key()
        self.assertIterInteresting([target, a_key, aab_key], [((b'aab',), b'new')], [target], [basis])

    def test_common_leaf(self):
        basis = self.get_map_key({})
        target1 = self.get_map_key({(b'aaa',): b'common'})
        target2 = self.get_map_key({(b'aaa',): b'common', (b'bbb',): b'new'})
        target3 = self.get_map_key({(b'aaa',): b'common', (b'aac',): b'other', (b'bbb',): b'new'})
        target1_map = CHKMap(self.get_chk_bytes(), target1)
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'common'\n", target1_map._dump_tree())
        target2_map = CHKMap(self.get_chk_bytes(), target2)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'common'\n  'b' LeafNode\n      ('bbb',) 'new'\n", target2_map._dump_tree())
        target3_map = CHKMap(self.get_chk_bytes(), target3)
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'common'\n    'aac' LeafNode\n      ('aac',) 'other'\n  'b' LeafNode\n      ('bbb',) 'new'\n", target3_map._dump_tree())
        aaa_key = target1_map._root_node.key()
        b_key = target2_map._root_node._items[b'b'].key()
        a_key = target3_map._root_node._items[b'a'].key()
        aac_key = target3_map._root_node._items[b'a']._items[b'aac'].key()
        self.assertIterInteresting([target1, target2, target3, a_key, aac_key, b_key], [((b'aaa',), b'common'), ((b'bbb',), b'new'), ((b'aac',), b'other')], [target1, target2, target3], [basis])
        self.assertIterInteresting([target2, target3, a_key, aac_key, b_key], [((b'bbb',), b'new'), ((b'aac',), b'other')], [target2, target3], [target1])
        self.assertIterInteresting([target1], [], [target1], [target3])

    def test_multiple_maps(self):
        basis1 = self.get_map_key({(b'aaa',): b'common', (b'aab',): b'basis1'})
        basis2 = self.get_map_key({(b'bbb',): b'common', (b'bbc',): b'basis2'})
        target1 = self.get_map_key({(b'aaa',): b'common', (b'aac',): b'target1', (b'bbb',): b'common'})
        target2 = self.get_map_key({(b'aaa',): b'common', (b'bba',): b'target2', (b'bbb',): b'common'})
        target1_map = CHKMap(self.get_chk_bytes(), target1)
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'common'\n    'aac' LeafNode\n      ('aac',) 'target1'\n  'b' LeafNode\n      ('bbb',) 'common'\n", target1_map._dump_tree())
        a_key = target1_map._root_node._items[b'a'].key()
        aac_key = target1_map._root_node._items[b'a']._items[b'aac'].key()
        target2_map = CHKMap(self.get_chk_bytes(), target2)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'common'\n  'b' InternalNode\n    'bba' LeafNode\n      ('bba',) 'target2'\n    'bbb' LeafNode\n      ('bbb',) 'common'\n", target2_map._dump_tree())
        b_key = target2_map._root_node._items[b'b'].key()
        bba_key = target2_map._root_node._items[b'b']._items[b'bba'].key()
        self.assertIterInteresting([target1, target2, a_key, aac_key, b_key, bba_key], [((b'aac',), b'target1'), ((b'bba',), b'target2')], [target1, target2], [basis1, basis2])

    def test_multiple_maps_overlapping_common_new(self):
        basis = self.get_map_key({(b'aaa',): b'left', (b'abb',): b'right', (b'ccc',): b'common'})
        left = self.get_map_key({(b'aaa',): b'left', (b'abb',): b'right', (b'ccc',): b'common', (b'ddd',): b'change'})
        right = self.get_map_key({(b'abb',): b'right'})
        basis_map = CHKMap(self.get_chk_bytes(), basis)
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aa' LeafNode\n      ('aaa',) 'left'\n    'ab' LeafNode\n      ('abb',) 'right'\n  'c' LeafNode\n      ('ccc',) 'common'\n", basis_map._dump_tree())
        left_map = CHKMap(self.get_chk_bytes(), left)
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aa' LeafNode\n      ('aaa',) 'left'\n    'ab' LeafNode\n      ('abb',) 'right'\n  'c' LeafNode\n      ('ccc',) 'common'\n  'd' LeafNode\n      ('ddd',) 'change'\n", left_map._dump_tree())
        l_d_key = left_map._root_node._items[b'd'].key()
        right_map = CHKMap(self.get_chk_bytes(), right)
        self.assertEqualDiff("'' LeafNode\n      ('abb',) 'right'\n", right_map._dump_tree())
        self.assertIterInteresting([right, left, l_d_key], [((b'ddd',), b'change')], [left, right], [basis])

    def test_multiple_maps_similar(self):
        basis = self.get_map_key({(b'aaa',): b'unchanged', (b'abb',): b'will change left', (b'caa',): b'unchanged', (b'cbb',): b'will change right'}, maximum_size=60)
        left = self.get_map_key({(b'aaa',): b'unchanged', (b'abb',): b'changed left', (b'caa',): b'unchanged', (b'cbb',): b'will change right'}, maximum_size=60)
        right = self.get_map_key({(b'aaa',): b'unchanged', (b'abb',): b'will change left', (b'caa',): b'unchanged', (b'cbb',): b'changed right'}, maximum_size=60)
        basis_map = CHKMap(self.get_chk_bytes(), basis)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'unchanged'\n      ('abb',) 'will change left'\n  'c' LeafNode\n      ('caa',) 'unchanged'\n      ('cbb',) 'will change right'\n", basis_map._dump_tree())
        left_map = CHKMap(self.get_chk_bytes(), left)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'unchanged'\n      ('abb',) 'changed left'\n  'c' LeafNode\n      ('caa',) 'unchanged'\n      ('cbb',) 'will change right'\n", left_map._dump_tree())
        l_a_key = left_map._root_node._items[b'a'].key()
        l_c_key = left_map._root_node._items[b'c'].key()
        right_map = CHKMap(self.get_chk_bytes(), right)
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'unchanged'\n      ('abb',) 'will change left'\n  'c' LeafNode\n      ('caa',) 'unchanged'\n      ('cbb',) 'changed right'\n", right_map._dump_tree())
        r_a_key = right_map._root_node._items[b'a'].key()
        r_c_key = right_map._root_node._items[b'c'].key()
        self.assertIterInteresting([right, left, l_a_key, r_c_key], [((b'abb',), b'changed left'), ((b'cbb',), b'changed right')], [left, right], [basis])