from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
class TestLeafNode(TestCaseWithStore):

    def test_current_size_empty(self):
        node = LeafNode()
        self.assertEqual(16, node._current_size())

    def test_current_size_size_changed(self):
        node = LeafNode()
        node.set_maximum_size(10)
        self.assertEqual(17, node._current_size())

    def test_current_size_width_changed(self):
        node = LeafNode()
        node._key_width = 10
        self.assertEqual(17, node._current_size())

    def test_current_size_items(self):
        node = LeafNode()
        base_size = node._current_size()
        node.map(None, (b'foo bar',), b'baz')
        self.assertEqual(base_size + 14, node._current_size())

    def test_deserialise_empty(self):
        node = LeafNode.deserialise(b'chkleaf:\n10\n1\n0\n\n', (b'sha1:1234',))
        self.assertEqual(0, len(node))
        self.assertEqual(10, node.maximum_size)
        self.assertEqual((b'sha1:1234',), node.key())
        self.assertIs(None, node._search_prefix)
        self.assertIs(None, node._common_serialised_prefix)

    def test_deserialise_items(self):
        node = LeafNode.deserialise(b'chkleaf:\n0\n1\n2\n\nfoo bar\x001\nbaz\nquux\x001\nblarh\n', (b'sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([((b'foo bar',), b'baz'), ((b'quux',), b'blarh')], sorted(node.iteritems(None)))

    def test_deserialise_item_with_null_width_1(self):
        node = LeafNode.deserialise(b'chkleaf:\n0\n1\n2\n\nfoo\x001\nbar\x00baz\nquux\x001\nblarh\n', (b'sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([((b'foo',), b'bar\x00baz'), ((b'quux',), b'blarh')], sorted(node.iteritems(None)))

    def test_deserialise_item_with_null_width_2(self):
        node = LeafNode.deserialise(b'chkleaf:\n0\n2\n2\n\nfoo\x001\x001\nbar\x00baz\nquux\x00\x001\nblarh\n', (b'sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([((b'foo', b'1'), b'bar\x00baz'), ((b'quux', b''), b'blarh')], sorted(node.iteritems(None)))

    def test_iteritems_selected_one_of_two_items(self):
        node = LeafNode.deserialise(b'chkleaf:\n0\n1\n2\n\nfoo bar\x001\nbaz\nquux\x001\nblarh\n', (b'sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([((b'quux',), b'blarh')], sorted(node.iteritems(None, [(b'quux',), (b'qaz',)])))

    def test_deserialise_item_with_common_prefix(self):
        node = LeafNode.deserialise(b'chkleaf:\n0\n2\n2\nfoo\x00\n1\x001\nbar\x00baz\n2\x001\nblarh\n', (b'sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([((b'foo', b'1'), b'bar\x00baz'), ((b'foo', b'2'), b'blarh')], sorted(node.iteritems(None)))
        self.assertIs(chk_map._unknown, node._search_prefix)
        self.assertEqual(b'foo\x00', node._common_serialised_prefix)

    def test_deserialise_multi_line(self):
        node = LeafNode.deserialise(b'chkleaf:\n0\n2\n2\nfoo\x00\n1\x002\nbar\nbaz\n2\x002\nblarh\n\n', (b'sha1:1234',))
        self.assertEqual(2, len(node))
        self.assertEqual([((b'foo', b'1'), b'bar\nbaz'), ((b'foo', b'2'), b'blarh\n')], sorted(node.iteritems(None)))
        self.assertIs(chk_map._unknown, node._search_prefix)
        self.assertEqual(b'foo\x00', node._common_serialised_prefix)

    def test_key_new(self):
        node = LeafNode()
        self.assertEqual(None, node.key())

    def test_key_after_map(self):
        node = LeafNode.deserialise(b'chkleaf:\n10\n1\n0\n\n', (b'sha1:1234',))
        node.map(None, (b'foo bar',), b'baz quux')
        self.assertEqual(None, node.key())

    def test_key_after_unmap(self):
        node = LeafNode.deserialise(b'chkleaf:\n0\n1\n2\n\nfoo bar\x001\nbaz\nquux\x001\nblarh\n', (b'sha1:1234',))
        node.unmap(None, (b'foo bar',))
        self.assertEqual(None, node.key())

    def test_map_exceeding_max_size_only_entry_new(self):
        node = LeafNode()
        node.set_maximum_size(10)
        result = node.map(None, (b'foo bar',), b'baz quux')
        self.assertEqual((b'foo bar', [(b'', node)]), result)
        self.assertTrue(10 < node._current_size())

    def test_map_exceeding_max_size_second_entry_early_difference_new(self):
        node = LeafNode()
        node.set_maximum_size(10)
        node.map(None, (b'foo bar',), b'baz quux')
        prefix, result = list(node.map(None, (b'blue',), b'red'))
        self.assertEqual(b'', prefix)
        self.assertEqual(2, len(result))
        split_chars = {result[0][0], result[1][0]}
        self.assertEqual({b'f', b'b'}, split_chars)
        nodes = dict(result)
        node = nodes[b'f']
        self.assertEqual({(b'foo bar',): b'baz quux'}, self.to_dict(node, None))
        self.assertEqual(10, node.maximum_size)
        self.assertEqual(1, node._key_width)
        node = nodes[b'b']
        self.assertEqual({(b'blue',): b'red'}, self.to_dict(node, None))
        self.assertEqual(10, node.maximum_size)
        self.assertEqual(1, node._key_width)

    def test_map_first(self):
        node = LeafNode()
        result = node.map(None, (b'foo bar',), b'baz quux')
        self.assertEqual((b'foo bar', [(b'', node)]), result)
        self.assertEqual({(b'foo bar',): b'baz quux'}, self.to_dict(node, None))
        self.assertEqual(1, len(node))

    def test_map_second(self):
        node = LeafNode()
        node.map(None, (b'foo bar',), b'baz quux')
        result = node.map(None, (b'bingo',), b'bango')
        self.assertEqual((b'', [(b'', node)]), result)
        self.assertEqual({(b'foo bar',): b'baz quux', (b'bingo',): b'bango'}, self.to_dict(node, None))
        self.assertEqual(2, len(node))

    def test_map_replacement(self):
        node = LeafNode()
        node.map(None, (b'foo bar',), b'baz quux')
        result = node.map(None, (b'foo bar',), b'bango')
        self.assertEqual((b'foo bar', [(b'', node)]), result)
        self.assertEqual({(b'foo bar',): b'bango'}, self.to_dict(node, None))
        self.assertEqual(1, len(node))

    def test_serialise_empty(self):
        store = self.get_chk_bytes()
        node = LeafNode()
        node.set_maximum_size(10)
        expected_key = (b'sha1:f34c3f0634ea3f85953dffa887620c0a5b1f4a51',)
        self.assertEqual([expected_key], list(node.serialise(store)))
        self.assertEqual(b'chkleaf:\n10\n1\n0\n\n', self.read_bytes(store, expected_key))
        self.assertEqual(expected_key, node.key())

    def test_serialise_items(self):
        store = self.get_chk_bytes()
        node = LeafNode()
        node.set_maximum_size(10)
        node.map(None, (b'foo bar',), b'baz quux')
        expected_key = (b'sha1:f89fac7edfc6bdb1b1b54a556012ff0c646ef5e0',)
        self.assertEqual(b'foo bar', node._common_serialised_prefix)
        self.assertEqual([expected_key], list(node.serialise(store)))
        self.assertEqual(b'chkleaf:\n10\n1\n1\nfoo bar\n\x001\nbaz quux\n', self.read_bytes(store, expected_key))
        self.assertEqual(expected_key, node.key())

    def test_unique_serialised_prefix_empty_new(self):
        node = LeafNode()
        self.assertIs(None, node._compute_search_prefix())

    def test_unique_serialised_prefix_one_item_new(self):
        node = LeafNode()
        node.map(None, (b'foo bar', b'baz'), b'baz quux')
        self.assertEqual(b'foo bar\x00baz', node._compute_search_prefix())

    def test_unmap_missing(self):
        node = LeafNode()
        self.assertRaises(KeyError, node.unmap, None, (b'foo bar',))

    def test_unmap_present(self):
        node = LeafNode()
        node.map(None, (b'foo bar',), b'baz quux')
        result = node.unmap(None, (b'foo bar',))
        self.assertEqual(node, result)
        self.assertEqual({}, self.to_dict(node, None))
        self.assertEqual(0, len(node))

    def test_map_maintains_common_prefixes(self):
        node = LeafNode()
        node._key_width = 2
        node.map(None, (b'foo bar', b'baz'), b'baz quux')
        self.assertEqual(b'foo bar\x00baz', node._search_prefix)
        self.assertEqual(b'foo bar\x00baz', node._common_serialised_prefix)
        node.map(None, (b'foo bar', b'bing'), b'baz quux')
        self.assertEqual(b'foo bar\x00b', node._search_prefix)
        self.assertEqual(b'foo bar\x00b', node._common_serialised_prefix)
        node.map(None, (b'fool', b'baby'), b'baz quux')
        self.assertEqual(b'foo', node._search_prefix)
        self.assertEqual(b'foo', node._common_serialised_prefix)
        node.map(None, (b'foo bar', b'baz'), b'replaced')
        self.assertEqual(b'foo', node._search_prefix)
        self.assertEqual(b'foo', node._common_serialised_prefix)
        node.map(None, (b'very', b'different'), b'value')
        self.assertEqual(b'', node._search_prefix)
        self.assertEqual(b'', node._common_serialised_prefix)

    def test_unmap_maintains_common_prefixes(self):
        node = LeafNode()
        node._key_width = 2
        node.map(None, (b'foo bar', b'baz'), b'baz quux')
        node.map(None, (b'foo bar', b'bing'), b'baz quux')
        node.map(None, (b'fool', b'baby'), b'baz quux')
        node.map(None, (b'very', b'different'), b'value')
        self.assertEqual(b'', node._search_prefix)
        self.assertEqual(b'', node._common_serialised_prefix)
        node.unmap(None, (b'very', b'different'))
        self.assertEqual(b'foo', node._search_prefix)
        self.assertEqual(b'foo', node._common_serialised_prefix)
        node.unmap(None, (b'fool', b'baby'))
        self.assertEqual(b'foo bar\x00b', node._search_prefix)
        self.assertEqual(b'foo bar\x00b', node._common_serialised_prefix)
        node.unmap(None, (b'foo bar', b'baz'))
        self.assertEqual(b'foo bar\x00bing', node._search_prefix)
        self.assertEqual(b'foo bar\x00bing', node._common_serialised_prefix)
        node.unmap(None, (b'foo bar', b'bing'))
        self.assertEqual(None, node._search_prefix)
        self.assertEqual(None, node._common_serialised_prefix)