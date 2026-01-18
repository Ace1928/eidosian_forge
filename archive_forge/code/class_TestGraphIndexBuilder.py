from ... import errors, tests, transport
from .. import index as _mod_index
class TestGraphIndexBuilder(tests.TestCaseWithMemoryTransport):

    def test_build_index_empty(self):
        builder = _mod_index.GraphIndexBuilder()
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=1\nlen=0\n\n', contents)

    def test_build_index_empty_two_element_keys(self):
        builder = _mod_index.GraphIndexBuilder(key_elements=2)
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=2\nlen=0\n\n', contents)

    def test_build_index_one_reference_list_empty(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=1)
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=1\nlen=0\n\n', contents)

    def test_build_index_two_reference_list_empty(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=2)
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=2\nkey_elements=1\nlen=0\n\n', contents)

    def test_build_index_one_node_no_refs(self):
        builder = _mod_index.GraphIndexBuilder()
        builder.add_node((b'akey',), b'data')
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=1\nlen=1\nakey\x00\x00\x00data\n\n', contents)

    def test_build_index_one_node_no_refs_accepts_empty_reflist(self):
        builder = _mod_index.GraphIndexBuilder()
        builder.add_node((b'akey',), b'data', ())
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=1\nlen=1\nakey\x00\x00\x00data\n\n', contents)

    def test_build_index_one_node_2_element_keys(self):
        builder = _mod_index.GraphIndexBuilder(key_elements=2)
        builder.add_node((b'akey', b'secondpart'), b'data')
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=2\nlen=1\nakey\x00secondpart\x00\x00\x00data\n\n', contents)

    def test_add_node_empty_value(self):
        builder = _mod_index.GraphIndexBuilder()
        builder.add_node((b'akey',), b'')
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=1\nlen=1\nakey\x00\x00\x00\n\n', contents)

    def test_build_index_nodes_sorted(self):
        builder = _mod_index.GraphIndexBuilder()
        builder.add_node((b'2002',), b'data')
        builder.add_node((b'2000',), b'data')
        builder.add_node((b'2001',), b'data')
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=1\nlen=3\n2000\x00\x00\x00data\n2001\x00\x00\x00data\n2002\x00\x00\x00data\n\n', contents)

    def test_build_index_2_element_key_nodes_sorted(self):
        builder = _mod_index.GraphIndexBuilder(key_elements=2)
        builder.add_node((b'2002', b'2002'), b'data')
        builder.add_node((b'2002', b'2000'), b'data')
        builder.add_node((b'2002', b'2001'), b'data')
        builder.add_node((b'2000', b'2002'), b'data')
        builder.add_node((b'2000', b'2000'), b'data')
        builder.add_node((b'2000', b'2001'), b'data')
        builder.add_node((b'2001', b'2002'), b'data')
        builder.add_node((b'2001', b'2000'), b'data')
        builder.add_node((b'2001', b'2001'), b'data')
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=2\nlen=9\n2000\x002000\x00\x00\x00data\n2000\x002001\x00\x00\x00data\n2000\x002002\x00\x00\x00data\n2001\x002000\x00\x00\x00data\n2001\x002001\x00\x00\x00data\n2001\x002002\x00\x00\x00data\n2002\x002000\x00\x00\x00data\n2002\x002001\x00\x00\x00data\n2002\x002002\x00\x00\x00data\n\n', contents)

    def test_build_index_reference_lists_are_included_one(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=1)
        builder.add_node((b'key',), b'data', ([],))
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=1\nlen=1\nkey\x00\x00\x00data\n\n', contents)

    def test_build_index_reference_lists_with_2_element_keys(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=1, key_elements=2)
        builder.add_node((b'key', b'key2'), b'data', ([],))
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=2\nlen=1\nkey\x00key2\x00\x00\x00data\n\n', contents)

    def test_build_index_reference_lists_are_included_two(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=2)
        builder.add_node((b'key',), b'data', ([], []))
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=2\nkey_elements=1\nlen=1\nkey\x00\x00\t\x00data\n\n', contents)

    def test_clear_cache(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=2)
        builder.clear_cache()

    def test_node_references_are_byte_offsets(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=1)
        builder.add_node((b'reference',), b'data', ([],))
        builder.add_node((b'key',), b'data', ([(b'reference',)],))
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=1\nlen=2\nkey\x00\x0072\x00data\nreference\x00\x00\x00data\n\n', contents)

    def test_node_references_are_cr_delimited(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=1)
        builder.add_node((b'reference',), b'data', ([],))
        builder.add_node((b'reference2',), b'data', ([],))
        builder.add_node((b'key',), b'data', ([(b'reference',), (b'reference2',)],))
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=1\nlen=3\nkey\x00\x00077\r094\x00data\nreference\x00\x00\x00data\nreference2\x00\x00\x00data\n\n', contents)

    def test_multiple_reference_lists_are_tab_delimited(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=2)
        builder.add_node((b'keference',), b'data', ([], []))
        builder.add_node((b'rey',), b'data', ([(b'keference',)], [(b'keference',)]))
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=2\nkey_elements=1\nlen=2\nkeference\x00\x00\t\x00data\nrey\x00\x0059\t59\x00data\n\n', contents)

    def test_add_node_referencing_missing_key_makes_absent(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=1)
        builder.add_node((b'rey',), b'data', ([(b'beference',), (b'aeference2',)],))
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=1\nlen=1\naeference2\x00a\x00\x00\nbeference\x00a\x00\x00\nrey\x00\x00074\r059\x00data\n\n', contents)

    def test_node_references_three_digits(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=1)
        references = [(b'%d' % val,) for val in range(8, -1, -1)]
        builder.add_node((b'2-key',), b'', (references,))
        stream = builder.finish()
        contents = stream.read()
        self.assertEqualDiff(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=1\nlen=1\n0\x00a\x00\x00\n1\x00a\x00\x00\n2\x00a\x00\x00\n2-key\x00\x00151\r145\r139\r133\r127\r121\r071\r065\r059\x00\n3\x00a\x00\x00\n4\x00a\x00\x00\n5\x00a\x00\x00\n6\x00a\x00\x00\n7\x00a\x00\x00\n8\x00a\x00\x00\n\n', contents)

    def test_absent_has_no_reference_overhead(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=2)
        builder.add_node((b'parent',), b'', ([(b'aail',), (b'zther',)], []))
        stream = builder.finish()
        contents = stream.read()
        self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=2\nkey_elements=1\nlen=1\naail\x00a\x00\x00\nparent\x00\x0059\r84\t\x00\nzther\x00a\x00\x00\n\n', contents)

    def test_add_node_bad_key(self):
        builder = _mod_index.GraphIndexBuilder()
        for bad_char in bytearray(b'\t\n\x0b\x0c\r\x00 '):
            self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'a%skey' % bytes([bad_char]),), b'data')
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (), b'data')
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, b'not-a-tuple', b'data')
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (), b'data')
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'primary', b'secondary'), b'data')
        builder = _mod_index.GraphIndexBuilder(key_elements=2)
        for bad_char in bytearray(b'\t\n\x0b\x0c\r\x00 '):
            self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'prefix', b'a%skey' % bytes([bad_char])), b'data')

    def test_add_node_bad_data(self):
        builder = _mod_index.GraphIndexBuilder()
        self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data\naa')
        self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data\x00aa')

    def test_add_node_bad_mismatched_ref_lists_length(self):
        builder = _mod_index.GraphIndexBuilder()
        self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa', ([],))
        builder = _mod_index.GraphIndexBuilder(reference_lists=1)
        self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa')
        self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa', ())
        self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa', ([], []))
        builder = _mod_index.GraphIndexBuilder(reference_lists=2)
        self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa')
        self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa', ([],))
        self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa', ([], [], []))

    def test_add_node_bad_key_in_reference_lists(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=1)
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', ([(b'a key',)],))
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', (['not-a-tuple'],))
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', ([()],))
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', ([(b'primary', b'secondary')],))
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', ([(b'agoodkey',), (b'that is a bad key',)],))
        builder = _mod_index.GraphIndexBuilder(reference_lists=2)
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', ([], ['a bad key']))

    def test_add_duplicate_key(self):
        builder = _mod_index.GraphIndexBuilder()
        builder.add_node((b'key',), b'data')
        self.assertRaises(_mod_index.BadIndexDuplicateKey, builder.add_node, (b'key',), b'data')

    def test_add_duplicate_key_2_elements(self):
        builder = _mod_index.GraphIndexBuilder(key_elements=2)
        builder.add_node((b'key', b'key'), b'data')
        self.assertRaises(_mod_index.BadIndexDuplicateKey, builder.add_node, (b'key', b'key'), b'data')

    def test_add_key_after_referencing_key(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=1)
        builder.add_node((b'key',), b'data', ([(b'reference',)],))
        builder.add_node((b'reference',), b'data', ([],))

    def test_add_key_after_referencing_key_2_elements(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=1, key_elements=2)
        builder.add_node((b'k', b'ey'), b'data', ([(b'reference', b'tokey')],))
        builder.add_node((b'reference', b'tokey'), b'data', ([],))

    def test_set_optimize(self):
        builder = _mod_index.GraphIndexBuilder(reference_lists=1, key_elements=2)
        builder.set_optimize(for_size=True)
        self.assertTrue(builder._optimize_for_size)
        builder.set_optimize(for_size=False)
        self.assertFalse(builder._optimize_for_size)