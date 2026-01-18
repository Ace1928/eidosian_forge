from ... import errors, tests, transport
from .. import index as _mod_index
class TestCombinedGraphIndex(tests.TestCaseWithMemoryTransport):

    def make_index(self, name, ref_lists=0, key_elements=1, nodes=[]):
        builder = _mod_index.GraphIndexBuilder(ref_lists, key_elements=key_elements)
        for key, value, references in nodes:
            builder.add_node(key, value, references)
        stream = builder.finish()
        trans = self.get_transport()
        size = trans.put_file(name, stream)
        return _mod_index.GraphIndex(trans, name, size)

    def make_combined_index_with_missing(self, missing=['1', '2']):
        """Create a CombinedGraphIndex which will have missing indexes.

        This creates a CGI which thinks it has 2 indexes, however they have
        been deleted. If CGI._reload_func() is called, then it will repopulate
        with a new index.

        :param missing: The underlying indexes to delete
        :return: (CombinedGraphIndex, reload_counter)
        """
        idx1 = self.make_index('1', nodes=[((b'1',), b'', ())])
        idx2 = self.make_index('2', nodes=[((b'2',), b'', ())])
        idx3 = self.make_index('3', nodes=[((b'1',), b'', ()), ((b'2',), b'', ())])
        reload_counter = [0, 0, 0]

        def reload():
            reload_counter[0] += 1
            new_indices = [idx3]
            if idx._indices == new_indices:
                reload_counter[2] += 1
                return False
            reload_counter[1] += 1
            idx._indices[:] = new_indices
            return True
        idx = _mod_index.CombinedGraphIndex([idx1, idx2], reload_func=reload)
        trans = self.get_transport()
        for fname in missing:
            trans.delete(fname)
        return (idx, reload_counter)

    def test_open_missing_index_no_error(self):
        trans = self.get_transport()
        idx1 = _mod_index.GraphIndex(trans, 'missing', 100)
        idx = _mod_index.CombinedGraphIndex([idx1])

    def test_add_index(self):
        idx = _mod_index.CombinedGraphIndex([])
        idx1 = self.make_index('name', 0, nodes=[((b'key',), b'', ())])
        idx.insert_index(0, idx1)
        self.assertEqual([(idx1, (b'key',), b'')], list(idx.iter_all_entries()))

    def test_clear_cache(self):
        log = []

        class ClearCacheProxy:

            def __init__(self, index):
                self._index = index

            def __getattr__(self, name):
                return getattr(self._index)

            def clear_cache(self):
                log.append(self._index)
                return self._index.clear_cache()
        idx = _mod_index.CombinedGraphIndex([])
        idx1 = self.make_index('name', 0, nodes=[((b'key',), b'', ())])
        idx.insert_index(0, ClearCacheProxy(idx1))
        idx2 = self.make_index('name', 0, nodes=[((b'key',), b'', ())])
        idx.insert_index(1, ClearCacheProxy(idx2))
        idx.clear_cache()
        self.assertEqual(sorted([idx1, idx2]), sorted(log))

    def test_iter_all_entries_empty(self):
        idx = _mod_index.CombinedGraphIndex([])
        self.assertEqual([], list(idx.iter_all_entries()))

    def test_iter_all_entries_children_empty(self):
        idx1 = self.make_index('name')
        idx = _mod_index.CombinedGraphIndex([idx1])
        self.assertEqual([], list(idx.iter_all_entries()))

    def test_iter_all_entries_simple(self):
        idx1 = self.make_index('name', nodes=[((b'name',), b'data', ())])
        idx = _mod_index.CombinedGraphIndex([idx1])
        self.assertEqual([(idx1, (b'name',), b'data')], list(idx.iter_all_entries()))

    def test_iter_all_entries_two_indices(self):
        idx1 = self.make_index('name1', nodes=[((b'name',), b'data', ())])
        idx2 = self.make_index('name2', nodes=[((b'2',), b'', ())])
        idx = _mod_index.CombinedGraphIndex([idx1, idx2])
        self.assertEqual([(idx1, (b'name',), b'data'), (idx2, (b'2',), b'')], list(idx.iter_all_entries()))

    def test_iter_entries_two_indices_dup_key(self):
        idx1 = self.make_index('name1', nodes=[((b'name',), b'data', ())])
        idx2 = self.make_index('name2', nodes=[((b'name',), b'data', ())])
        idx = _mod_index.CombinedGraphIndex([idx1, idx2])
        self.assertEqual([(idx1, (b'name',), b'data')], list(idx.iter_entries([(b'name',)])))

    def test_iter_all_entries_two_indices_dup_key(self):
        idx1 = self.make_index('name1', nodes=[((b'name',), b'data', ())])
        idx2 = self.make_index('name2', nodes=[((b'name',), b'data', ())])
        idx = _mod_index.CombinedGraphIndex([idx1, idx2])
        self.assertEqual([(idx1, (b'name',), b'data')], list(idx.iter_all_entries()))

    def test_iter_key_prefix_2_key_element_refs(self):
        idx1 = self.make_index('1', 1, key_elements=2, nodes=[((b'name', b'fin1'), b'data', ([(b'ref', b'erence')],))])
        idx2 = self.make_index('2', 1, key_elements=2, nodes=[((b'name', b'fin2'), b'beta', ([],)), ((b'ref', b'erence'), b'refdata', ([],))])
        idx = _mod_index.CombinedGraphIndex([idx1, idx2])
        self.assertEqual({(idx1, (b'name', b'fin1'), b'data', (((b'ref', b'erence'),),)), (idx2, (b'ref', b'erence'), b'refdata', ((),))}, set(idx.iter_entries_prefix([(b'name', b'fin1'), (b'ref', b'erence')])))
        self.assertEqual({(idx1, (b'name', b'fin1'), b'data', (((b'ref', b'erence'),),)), (idx2, (b'name', b'fin2'), b'beta', ((),))}, set(idx.iter_entries_prefix([(b'name', None)])))

    def test_iter_nothing_empty(self):
        idx = _mod_index.CombinedGraphIndex([])
        self.assertEqual([], list(idx.iter_entries([])))

    def test_iter_nothing_children_empty(self):
        idx1 = self.make_index('name')
        idx = _mod_index.CombinedGraphIndex([idx1])
        self.assertEqual([], list(idx.iter_entries([])))

    def test_iter_all_keys(self):
        idx1 = self.make_index('1', 1, nodes=[((b'name',), b'data', ([(b'ref',)],))])
        idx2 = self.make_index('2', 1, nodes=[((b'ref',), b'refdata', ((),))])
        idx = _mod_index.CombinedGraphIndex([idx1, idx2])
        self.assertEqual({(idx1, (b'name',), b'data', (((b'ref',),),)), (idx2, (b'ref',), b'refdata', ((),))}, set(idx.iter_entries([(b'name',), (b'ref',)])))

    def test_iter_all_keys_dup_entry(self):
        idx1 = self.make_index('1', 1, nodes=[((b'name',), b'data', ([(b'ref',)],)), ((b'ref',), b'refdata', ([],))])
        idx2 = self.make_index('2', 1, nodes=[((b'ref',), b'refdata', ([],))])
        idx = _mod_index.CombinedGraphIndex([idx1, idx2])
        self.assertEqual({(idx1, (b'name',), b'data', (((b'ref',),),)), (idx1, (b'ref',), b'refdata', ((),))}, set(idx.iter_entries([(b'name',), (b'ref',)])))

    def test_iter_missing_entry_empty(self):
        idx = _mod_index.CombinedGraphIndex([])
        self.assertEqual([], list(idx.iter_entries([('a',)])))

    def test_iter_missing_entry_one_index(self):
        idx1 = self.make_index('1')
        idx = _mod_index.CombinedGraphIndex([idx1])
        self.assertEqual([], list(idx.iter_entries([(b'a',)])))

    def test_iter_missing_entry_two_index(self):
        idx1 = self.make_index('1')
        idx2 = self.make_index('2')
        idx = _mod_index.CombinedGraphIndex([idx1, idx2])
        self.assertEqual([], list(idx.iter_entries([('a',)])))

    def test_iter_entry_present_one_index_only(self):
        idx1 = self.make_index('1', nodes=[((b'key',), b'', ())])
        idx2 = self.make_index('2', nodes=[])
        idx = _mod_index.CombinedGraphIndex([idx1, idx2])
        self.assertEqual([(idx1, (b'key',), b'')], list(idx.iter_entries([(b'key',)])))
        idx = _mod_index.CombinedGraphIndex([idx2, idx1])
        self.assertEqual([(idx1, (b'key',), b'')], list(idx.iter_entries([(b'key',)])))

    def test_key_count_empty(self):
        idx1 = self.make_index('1', nodes=[])
        idx2 = self.make_index('2', nodes=[])
        idx = _mod_index.CombinedGraphIndex([idx1, idx2])
        self.assertEqual(0, idx.key_count())

    def test_key_count_sums_index_keys(self):
        idx1 = self.make_index('1', nodes=[((b'1',), b'', ()), ((b'2',), b'', ())])
        idx2 = self.make_index('2', nodes=[((b'1',), b'', ())])
        idx = _mod_index.CombinedGraphIndex([idx1, idx2])
        self.assertEqual(3, idx.key_count())

    def test_validate_bad_child_index_errors(self):
        trans = self.get_transport()
        trans.put_bytes('name', b'not an index\n')
        idx1 = _mod_index.GraphIndex(trans, 'name', 13)
        idx = _mod_index.CombinedGraphIndex([idx1])
        self.assertRaises(_mod_index.BadIndexFormatSignature, idx.validate)

    def test_validate_empty(self):
        idx = _mod_index.CombinedGraphIndex([])
        idx.validate()

    def test_key_count_reloads(self):
        idx, reload_counter = self.make_combined_index_with_missing()
        self.assertEqual(2, idx.key_count())
        self.assertEqual([1, 1, 0], reload_counter)

    def test_key_count_no_reload(self):
        idx, reload_counter = self.make_combined_index_with_missing()
        idx._reload_func = None
        self.assertRaises(transport.NoSuchFile, idx.key_count)

    def test_key_count_reloads_and_fails(self):
        idx, reload_counter = self.make_combined_index_with_missing(['1', '2', '3'])
        self.assertRaises(transport.NoSuchFile, idx.key_count)
        self.assertEqual([2, 1, 1], reload_counter)

    def test_iter_entries_reloads(self):
        index, reload_counter = self.make_combined_index_with_missing()
        result = list(index.iter_entries([(b'1',), (b'2',), (b'3',)]))
        index3 = index._indices[0]
        self.assertEqual({(index3, (b'1',), b''), (index3, (b'2',), b'')}, set(result))
        self.assertEqual([1, 1, 0], reload_counter)

    def test_iter_entries_reloads_midway(self):
        index, reload_counter = self.make_combined_index_with_missing(['2'])
        index1, index2 = index._indices
        result = list(index.iter_entries([(b'1',), (b'2',), (b'3',)]))
        index3 = index._indices[0]
        self.assertEqual([(index1, (b'1',), b''), (index3, (b'2',), b'')], result)
        self.assertEqual([1, 1, 0], reload_counter)

    def test_iter_entries_no_reload(self):
        index, reload_counter = self.make_combined_index_with_missing()
        index._reload_func = None
        self.assertListRaises(transport.NoSuchFile, index.iter_entries, [('3',)])

    def test_iter_entries_reloads_and_fails(self):
        index, reload_counter = self.make_combined_index_with_missing(['1', '2', '3'])
        self.assertListRaises(transport.NoSuchFile, index.iter_entries, [('3',)])
        self.assertEqual([2, 1, 1], reload_counter)

    def test_iter_all_entries_reloads(self):
        index, reload_counter = self.make_combined_index_with_missing()
        result = list(index.iter_all_entries())
        index3 = index._indices[0]
        self.assertEqual({(index3, (b'1',), b''), (index3, (b'2',), b'')}, set(result))
        self.assertEqual([1, 1, 0], reload_counter)

    def test_iter_all_entries_reloads_midway(self):
        index, reload_counter = self.make_combined_index_with_missing(['2'])
        index1, index2 = index._indices
        result = list(index.iter_all_entries())
        index3 = index._indices[0]
        self.assertEqual([(index1, (b'1',), b''), (index3, (b'2',), b'')], result)
        self.assertEqual([1, 1, 0], reload_counter)

    def test_iter_all_entries_no_reload(self):
        index, reload_counter = self.make_combined_index_with_missing()
        index._reload_func = None
        self.assertListRaises(transport.NoSuchFile, index.iter_all_entries)

    def test_iter_all_entries_reloads_and_fails(self):
        index, reload_counter = self.make_combined_index_with_missing(['1', '2', '3'])
        self.assertListRaises(transport.NoSuchFile, index.iter_all_entries)

    def test_iter_entries_prefix_reloads(self):
        index, reload_counter = self.make_combined_index_with_missing()
        result = list(index.iter_entries_prefix([(b'1',)]))
        index3 = index._indices[0]
        self.assertEqual([(index3, (b'1',), b'')], result)
        self.assertEqual([1, 1, 0], reload_counter)

    def test_iter_entries_prefix_reloads_midway(self):
        index, reload_counter = self.make_combined_index_with_missing(['2'])
        index1, index2 = index._indices
        result = list(index.iter_entries_prefix([(b'1',)]))
        index3 = index._indices[0]
        self.assertEqual([(index1, (b'1',), b'')], result)
        self.assertEqual([1, 1, 0], reload_counter)

    def test_iter_entries_prefix_no_reload(self):
        index, reload_counter = self.make_combined_index_with_missing()
        index._reload_func = None
        self.assertListRaises(transport.NoSuchFile, index.iter_entries_prefix, [(b'1',)])

    def test_iter_entries_prefix_reloads_and_fails(self):
        index, reload_counter = self.make_combined_index_with_missing(['1', '2', '3'])
        self.assertListRaises(transport.NoSuchFile, index.iter_entries_prefix, [(b'1',)])

    def make_index_with_simple_nodes(self, name, num_nodes=1):
        """Make an index named after 'name', with keys named after 'name' too.

        Nodes will have a value of '' and no references.
        """
        nodes = [(('index-{}-key-{}'.format(name, n).encode('ascii'),), b'', ()) for n in range(1, num_nodes + 1)]
        return self.make_index('index-%s' % name, 0, nodes=nodes)

    def test_reorder_after_iter_entries(self):
        idx = _mod_index.CombinedGraphIndex([])
        idx.insert_index(0, self.make_index_with_simple_nodes('1'), b'1')
        idx.insert_index(1, self.make_index_with_simple_nodes('2'), b'2')
        idx.insert_index(2, self.make_index_with_simple_nodes('3'), b'3')
        idx.insert_index(3, self.make_index_with_simple_nodes('4'), b'4')
        idx1, idx2, idx3, idx4 = idx._indices
        self.assertLength(2, list(idx.iter_entries([(b'index-4-key-1',), (b'index-2-key-1',)])))
        self.assertEqual([idx2, idx4, idx1, idx3], idx._indices)
        self.assertEqual([b'2', b'4', b'1', b'3'], idx._index_names)

    def test_reorder_propagates_to_siblings(self):
        cgi1 = _mod_index.CombinedGraphIndex([])
        cgi2 = _mod_index.CombinedGraphIndex([])
        cgi1.insert_index(0, self.make_index_with_simple_nodes('1-1'), 'one')
        cgi1.insert_index(1, self.make_index_with_simple_nodes('1-2'), 'two')
        cgi2.insert_index(0, self.make_index_with_simple_nodes('2-1'), 'one')
        cgi2.insert_index(1, self.make_index_with_simple_nodes('2-2'), 'two')
        index2_1, index2_2 = cgi2._indices
        cgi1.set_sibling_indices([cgi2])
        list(cgi1.iter_entries([(b'index-1-2-key-1',)]))
        self.assertEqual([index2_2, index2_1], cgi2._indices)
        self.assertEqual(['two', 'one'], cgi2._index_names)

    def test_validate_reloads(self):
        idx, reload_counter = self.make_combined_index_with_missing()
        idx.validate()
        self.assertEqual([1, 1, 0], reload_counter)

    def test_validate_reloads_midway(self):
        idx, reload_counter = self.make_combined_index_with_missing(['2'])
        idx.validate()

    def test_validate_no_reload(self):
        idx, reload_counter = self.make_combined_index_with_missing()
        idx._reload_func = None
        self.assertRaises(transport.NoSuchFile, idx.validate)

    def test_validate_reloads_and_fails(self):
        idx, reload_counter = self.make_combined_index_with_missing(['1', '2', '3'])
        self.assertRaises(transport.NoSuchFile, idx.validate)

    def test_find_ancestors_across_indexes(self):
        key1 = (b'key-1',)
        key2 = (b'key-2',)
        key3 = (b'key-3',)
        key4 = (b'key-4',)
        index1 = self.make_index('12', ref_lists=1, nodes=[(key1, b'value', ([],)), (key2, b'value', ([key1],))])
        index2 = self.make_index('34', ref_lists=1, nodes=[(key3, b'value', ([key2],)), (key4, b'value', ([key3],))])
        c_index = _mod_index.CombinedGraphIndex([index1, index2])
        parent_map, missing_keys = c_index.find_ancestry([key1], 0)
        self.assertEqual({key1: ()}, parent_map)
        self.assertEqual(set(), missing_keys)
        parent_map, missing_keys = c_index.find_ancestry([key3], 0)
        self.assertEqual({key1: (), key2: (key1,), key3: (key2,)}, parent_map)
        self.assertEqual(set(), missing_keys)

    def test_find_ancestors_missing_keys(self):
        key1 = (b'key-1',)
        key2 = (b'key-2',)
        key3 = (b'key-3',)
        key4 = (b'key-4',)
        index1 = self.make_index('12', ref_lists=1, nodes=[(key1, b'value', ([],)), (key2, b'value', ([key1],))])
        index2 = self.make_index('34', ref_lists=1, nodes=[(key3, b'value', ([key2],))])
        c_index = _mod_index.CombinedGraphIndex([index1, index2])
        parent_map, missing_keys = c_index.find_ancestry([key4], 0)
        self.assertEqual({}, parent_map)
        self.assertEqual({key4}, missing_keys)

    def test_find_ancestors_no_indexes(self):
        c_index = _mod_index.CombinedGraphIndex([])
        key1 = (b'key-1',)
        parent_map, missing_keys = c_index.find_ancestry([key1], 0)
        self.assertEqual({}, parent_map)
        self.assertEqual({key1}, missing_keys)

    def test_find_ancestors_ghost_parent(self):
        key1 = (b'key-1',)
        key2 = (b'key-2',)
        key3 = (b'key-3',)
        key4 = (b'key-4',)
        index1 = self.make_index('12', ref_lists=1, nodes=[(key1, b'value', ([],)), (key2, b'value', ([key1],))])
        index2 = self.make_index('34', ref_lists=1, nodes=[(key4, b'value', ([key2, key3],))])
        c_index = _mod_index.CombinedGraphIndex([index1, index2])
        parent_map, missing_keys = c_index.find_ancestry([key4], 0)
        self.assertEqual({key4: (key2, key3), key2: (key1,), key1: ()}, parent_map)
        self.assertEqual({key3}, missing_keys)

    def test__find_ancestors_empty_index(self):
        idx = self.make_index('test', ref_lists=1, key_elements=1, nodes=[])
        parent_map = {}
        missing_keys = set()
        search_keys = idx._find_ancestors([(b'one',), (b'two',)], 0, parent_map, missing_keys)
        self.assertEqual(set(), search_keys)
        self.assertEqual({}, parent_map)
        self.assertEqual({(b'one',), (b'two',)}, missing_keys)