import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
class TestBTreeNodes(BTreeTestCase):
    scenarios = btreeparser_scenarios()

    def setUp(self):
        super().setUp()
        self.overrideAttr(btree_index, '_btree_serializer', self.parse_btree)

    def test_LeafNode_1_0(self):
        node_bytes = b'type=leaf\n0000000000000000000000000000000000000000\x00\x00value:0\n1111111111111111111111111111111111111111\x00\x00value:1\n2222222222222222222222222222222222222222\x00\x00value:2\n3333333333333333333333333333333333333333\x00\x00value:3\n4444444444444444444444444444444444444444\x00\x00value:4\n'
        node = btree_index._LeafNode(node_bytes, 1, 0)
        self.assertEqual({(b'0000000000000000000000000000000000000000',): (b'value:0', ()), (b'1111111111111111111111111111111111111111',): (b'value:1', ()), (b'2222222222222222222222222222222222222222',): (b'value:2', ()), (b'3333333333333333333333333333333333333333',): (b'value:3', ()), (b'4444444444444444444444444444444444444444',): (b'value:4', ())}, dict(node.all_items()))

    def test_LeafNode_2_2(self):
        node_bytes = b'type=leaf\n00\x0000\x00\t00\x00ref00\x00value:0\n00\x0011\x0000\x00ref00\t00\x00ref00\r01\x00ref01\x00value:1\n11\x0033\x0011\x00ref22\t11\x00ref22\r11\x00ref22\x00value:3\n11\x0044\x00\t11\x00ref00\x00value:4\n'
        node = btree_index._LeafNode(node_bytes, 2, 2)
        self.assertEqual({(b'00', b'00'): (b'value:0', ((), ((b'00', b'ref00'),))), (b'00', b'11'): (b'value:1', (((b'00', b'ref00'),), ((b'00', b'ref00'), (b'01', b'ref01')))), (b'11', b'33'): (b'value:3', (((b'11', b'ref22'),), ((b'11', b'ref22'), (b'11', b'ref22')))), (b'11', b'44'): (b'value:4', ((), ((b'11', b'ref00'),)))}, dict(node.all_items()))

    def test_InternalNode_1(self):
        node_bytes = b'type=internal\noffset=1\n0000000000000000000000000000000000000000\n1111111111111111111111111111111111111111\n2222222222222222222222222222222222222222\n3333333333333333333333333333333333333333\n4444444444444444444444444444444444444444\n'
        node = btree_index._InternalNode(node_bytes)
        self.assertEqual([(b'0000000000000000000000000000000000000000',), (b'1111111111111111111111111111111111111111',), (b'2222222222222222222222222222222222222222',), (b'3333333333333333333333333333333333333333',), (b'4444444444444444444444444444444444444444',)], node.keys)
        self.assertEqual(1, node.offset)

    def assertFlattened(self, expected, key, value, refs):
        flat_key, flat_line = self.parse_btree._flatten_node((None, key, value, refs), bool(refs))
        self.assertEqual(b'\x00'.join(key), flat_key)
        self.assertEqual(expected, flat_line)

    def test__flatten_node(self):
        self.assertFlattened(b'key\x00\x00value\n', (b'key',), b'value', [])
        self.assertFlattened(b'key\x00tuple\x00\x00value str\n', (b'key', b'tuple'), b'value str', [])
        self.assertFlattened(b'key\x00tuple\x00triple\x00\x00value str\n', (b'key', b'tuple', b'triple'), b'value str', [])
        self.assertFlattened(b'k\x00t\x00s\x00ref\x00value str\n', (b'k', b't', b's'), b'value str', [[(b'ref',)]])
        self.assertFlattened(b'key\x00tuple\x00ref\x00key\x00value str\n', (b'key', b'tuple'), b'value str', [[(b'ref', b'key')]])
        self.assertFlattened(b'00\x0000\x00\t00\x00ref00\x00value:0\n', (b'00', b'00'), b'value:0', ((), ((b'00', b'ref00'),)))
        self.assertFlattened(b'00\x0011\x0000\x00ref00\t00\x00ref00\r01\x00ref01\x00value:1\n', (b'00', b'11'), b'value:1', (((b'00', b'ref00'),), ((b'00', b'ref00'), (b'01', b'ref01'))))
        self.assertFlattened(b'11\x0033\x0011\x00ref22\t11\x00ref22\r11\x00ref22\x00value:3\n', (b'11', b'33'), b'value:3', (((b'11', b'ref22'),), ((b'11', b'ref22'), (b'11', b'ref22'))))
        self.assertFlattened(b'11\x0044\x00\t11\x00ref00\x00value:4\n', (b'11', b'44'), b'value:4', ((), ((b'11', b'ref00'),)))