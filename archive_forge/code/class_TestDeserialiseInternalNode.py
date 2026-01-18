from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
class TestDeserialiseInternalNode(tests.TestCase):
    module = None

    def assertDeserialiseErrors(self, text):
        self.assertRaises((ValueError, IndexError), self.module._deserialise_internal_node, text, stuple(b'not-a-real-sha'))

    def test_raises_on_non_internal(self):
        self.assertDeserialiseErrors(b'')
        self.assertDeserialiseErrors(b'short\n')
        self.assertDeserialiseErrors(b'chknotnode:\n')
        self.assertDeserialiseErrors(b'chknode:x\n')
        self.assertDeserialiseErrors(b'chknode:\n')
        self.assertDeserialiseErrors(b'chknode:\nnotint\n')
        self.assertDeserialiseErrors(b'chknode:\n10\n')
        self.assertDeserialiseErrors(b'chknode:\n10\n256\n')
        self.assertDeserialiseErrors(b'chknode:\n10\n256\n10\n')
        self.assertDeserialiseErrors(b'chknode:\n10\n256\n0\n1\nfo')

    def test_deserialise_one(self):
        node = self.module._deserialise_internal_node(b'chknode:\n10\n1\n1\n\na\x00sha1:abcd\n', stuple(b'sha1:1234'))
        self.assertIsInstance(node, chk_map.InternalNode)
        self.assertEqual(1, len(node))
        self.assertEqual(10, node.maximum_size)
        self.assertEqual((b'sha1:1234',), node.key())
        self.assertEqual(b'', node._search_prefix)
        self.assertEqual({b'a': (b'sha1:abcd',)}, node._items)

    def test_deserialise_with_prefix(self):
        node = self.module._deserialise_internal_node(b'chknode:\n10\n1\n1\npref\na\x00sha1:abcd\n', stuple(b'sha1:1234'))
        self.assertIsInstance(node, chk_map.InternalNode)
        self.assertEqual(1, len(node))
        self.assertEqual(10, node.maximum_size)
        self.assertEqual((b'sha1:1234',), node.key())
        self.assertEqual(b'pref', node._search_prefix)
        self.assertEqual({b'prefa': (b'sha1:abcd',)}, node._items)
        node = self.module._deserialise_internal_node(b'chknode:\n10\n1\n1\npref\n\x00sha1:abcd\n', stuple(b'sha1:1234'))
        self.assertIsInstance(node, chk_map.InternalNode)
        self.assertEqual(1, len(node))
        self.assertEqual(10, node.maximum_size)
        self.assertEqual((b'sha1:1234',), node.key())
        self.assertEqual(b'pref', node._search_prefix)
        self.assertEqual({b'pref': (b'sha1:abcd',)}, node._items)

    def test_deserialise_pref_with_null(self):
        node = self.module._deserialise_internal_node(b'chknode:\n10\n1\n1\npref\x00fo\n\x00sha1:abcd\n', stuple(b'sha1:1234'))
        self.assertIsInstance(node, chk_map.InternalNode)
        self.assertEqual(1, len(node))
        self.assertEqual(10, node.maximum_size)
        self.assertEqual((b'sha1:1234',), node.key())
        self.assertEqual(b'pref\x00fo', node._search_prefix)
        self.assertEqual({b'pref\x00fo': (b'sha1:abcd',)}, node._items)

    def test_deserialise_with_null_pref(self):
        node = self.module._deserialise_internal_node(b'chknode:\n10\n1\n1\npref\x00fo\n\x00\x00sha1:abcd\n', stuple(b'sha1:1234'))
        self.assertIsInstance(node, chk_map.InternalNode)
        self.assertEqual(1, len(node))
        self.assertEqual(10, node.maximum_size)
        self.assertEqual((b'sha1:1234',), node.key())
        self.assertEqual(b'pref\x00fo', node._search_prefix)
        self.assertEqual({b'pref\x00fo\x00': (b'sha1:abcd',)}, node._items)