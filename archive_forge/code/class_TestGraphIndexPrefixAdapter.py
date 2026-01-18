from ... import errors, tests, transport
from .. import index as _mod_index
class TestGraphIndexPrefixAdapter(tests.TestCaseWithMemoryTransport):

    def make_index(self, ref_lists=1, key_elements=2, nodes=[], add_callback=False):
        result = _mod_index.InMemoryGraphIndex(ref_lists, key_elements=key_elements)
        result.add_nodes(nodes)
        if add_callback:
            add_nodes_callback = result.add_nodes
        else:
            add_nodes_callback = None
        adapter = _mod_index.GraphIndexPrefixAdapter(result, (b'prefix',), key_elements - 1, add_nodes_callback=add_nodes_callback)
        return (result, adapter)

    def test_add_node(self):
        index, adapter = self.make_index(add_callback=True)
        adapter.add_node((b'key',), b'value', (((b'ref',),),))
        self.assertEqual({(index, (b'prefix', b'key'), b'value', (((b'prefix', b'ref'),),))}, set(index.iter_all_entries()))

    def test_add_nodes(self):
        index, adapter = self.make_index(add_callback=True)
        adapter.add_nodes((((b'key',), b'value', (((b'ref',),),)), ((b'key2',), b'value2', ((),))))
        self.assertEqual({(index, (b'prefix', b'key2'), b'value2', ((),)), (index, (b'prefix', b'key'), b'value', (((b'prefix', b'ref'),),))}, set(index.iter_all_entries()))

    def test_construct(self):
        idx = _mod_index.InMemoryGraphIndex()
        adapter = _mod_index.GraphIndexPrefixAdapter(idx, (b'prefix',), 1)

    def test_construct_with_callback(self):
        idx = _mod_index.InMemoryGraphIndex()
        adapter = _mod_index.GraphIndexPrefixAdapter(idx, (b'prefix',), 1, idx.add_nodes)

    def test_iter_all_entries_cross_prefix_map_errors(self):
        index, adapter = self.make_index(nodes=[((b'prefix', b'key1'), b'data1', (((b'prefixaltered', b'key2'),),))])
        self.assertRaises(_mod_index.BadIndexData, list, adapter.iter_all_entries())

    def test_iter_all_entries(self):
        index, adapter = self.make_index(nodes=[((b'notprefix', b'key1'), b'data', ((),)), ((b'prefix', b'key1'), b'data1', ((),)), ((b'prefix', b'key2'), b'data2', (((b'prefix', b'key1'),),))])
        self.assertEqual({(index, (b'key1',), b'data1', ((),)), (index, (b'key2',), b'data2', (((b'key1',),),))}, set(adapter.iter_all_entries()))

    def test_iter_entries(self):
        index, adapter = self.make_index(nodes=[((b'notprefix', b'key1'), b'data', ((),)), ((b'prefix', b'key1'), b'data1', ((),)), ((b'prefix', b'key2'), b'data2', (((b'prefix', b'key1'),),))])
        self.assertEqual({(index, (b'key1',), b'data1', ((),)), (index, (b'key2',), b'data2', (((b'key1',),),))}, set(adapter.iter_entries([(b'key1',), (b'key2',)])))
        self.assertEqual({(index, (b'key1',), b'data1', ((),))}, set(adapter.iter_entries([(b'key1',)])))
        self.assertEqual(set(), set(adapter.iter_entries([(b'key3',)])))

    def test_iter_entries_prefix(self):
        index, adapter = self.make_index(key_elements=3, nodes=[((b'notprefix', b'foo', b'key1'), b'data', ((),)), ((b'prefix', b'prefix2', b'key1'), b'data1', ((),)), ((b'prefix', b'prefix2', b'key2'), b'data2', (((b'prefix', b'prefix2', b'key1'),),))])
        self.assertEqual({(index, (b'prefix2', b'key1'), b'data1', ((),)), (index, (b'prefix2', b'key2'), b'data2', (((b'prefix2', b'key1'),),))}, set(adapter.iter_entries_prefix([(b'prefix2', None)])))

    def test_key_count_no_matching_keys(self):
        index, adapter = self.make_index(nodes=[((b'notprefix', b'key1'), b'data', ((),))])
        self.assertEqual(0, adapter.key_count())

    def test_key_count_some_keys(self):
        index, adapter = self.make_index(nodes=[((b'notprefix', b'key1'), b'data', ((),)), ((b'prefix', b'key1'), b'data1', ((),)), ((b'prefix', b'key2'), b'data2', (((b'prefix', b'key1'),),))])
        self.assertEqual(2, adapter.key_count())

    def test_validate(self):
        index, adapter = self.make_index()
        calls = []

        def validate():
            calls.append('called')
        index.validate = validate
        adapter.validate()
        self.assertEqual(['called'], calls)