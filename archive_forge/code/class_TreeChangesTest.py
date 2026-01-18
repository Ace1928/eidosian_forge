from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
class TreeChangesTest(DiffTestCase):

    def setUp(self):
        super().setUp()
        self.detector = RenameDetector(self.store)

    def assertMergeFails(self, merge_entries, name, mode, sha):
        t = Tree()
        t[name] = (mode, sha)
        self.assertRaises((TypeError, ValueError), merge_entries, '', t, t)

    def _do_test_merge_entries(self, merge_entries):
        blob_a1 = make_object(Blob, data=b'a1')
        blob_a2 = make_object(Blob, data=b'a2')
        blob_b1 = make_object(Blob, data=b'b1')
        blob_c2 = make_object(Blob, data=b'c2')
        tree1 = self.commit_tree([(b'a', blob_a1, 33188), (b'b', blob_b1, 33261)])
        tree2 = self.commit_tree([(b'a', blob_a2, 33188), (b'c', blob_c2, 33261)])
        self.assertEqual([], merge_entries(b'', self.empty_tree, self.empty_tree))
        self.assertEqual([((None, None, None), (b'a', 33188, blob_a1.id)), ((None, None, None), (b'b', 33261, blob_b1.id))], merge_entries(b'', self.empty_tree, tree1))
        self.assertEqual([((None, None, None), (b'x/a', 33188, blob_a1.id)), ((None, None, None), (b'x/b', 33261, blob_b1.id))], merge_entries(b'x', self.empty_tree, tree1))
        self.assertEqual([((b'a', 33188, blob_a2.id), (None, None, None)), ((b'c', 33261, blob_c2.id), (None, None, None))], merge_entries(b'', tree2, self.empty_tree))
        self.assertEqual([((b'a', 33188, blob_a1.id), (b'a', 33188, blob_a2.id)), ((b'b', 33261, blob_b1.id), (None, None, None)), ((None, None, None), (b'c', 33261, blob_c2.id))], merge_entries(b'', tree1, tree2))
        self.assertEqual([((b'a', 33188, blob_a2.id), (b'a', 33188, blob_a1.id)), ((None, None, None), (b'b', 33261, blob_b1.id)), ((b'c', 33261, blob_c2.id), (None, None, None))], merge_entries(b'', tree2, tree1))
        self.assertMergeFails(merge_entries, 3735928559, 33188, '1' * 40)
        self.assertMergeFails(merge_entries, b'a', b'deadbeef', '1' * 40)
        self.assertMergeFails(merge_entries, b'a', 33188, 3735928559)
    test_merge_entries = functest_builder(_do_test_merge_entries, _merge_entries_py)
    test_merge_entries_extension = ext_functest_builder(_do_test_merge_entries, _merge_entries)

    def _do_test_is_tree(self, is_tree):
        self.assertFalse(is_tree(TreeEntry(None, None, None)))
        self.assertFalse(is_tree(TreeEntry(b'a', 33188, b'a' * 40)))
        self.assertFalse(is_tree(TreeEntry(b'a', 33261, b'a' * 40)))
        self.assertFalse(is_tree(TreeEntry(b'a', 40960, b'a' * 40)))
        self.assertTrue(is_tree(TreeEntry(b'a', 16384, b'a' * 40)))
        self.assertRaises(TypeError, is_tree, TreeEntry(b'a', b'x', b'a' * 40))
        self.assertRaises(AttributeError, is_tree, 1234)
    test_is_tree = functest_builder(_do_test_is_tree, _is_tree_py)
    test_is_tree_extension = ext_functest_builder(_do_test_is_tree, _is_tree)

    def assertChangesEqual(self, expected, tree1, tree2, **kwargs):
        actual = list(tree_changes(self.store, tree1.id, tree2.id, **kwargs))
        self.assertEqual(expected, actual)

    def test_tree_changes_empty(self):
        self.assertChangesEqual([], self.empty_tree, self.empty_tree)

    def test_tree_changes_no_changes(self):
        blob = make_object(Blob, data=b'blob')
        tree = self.commit_tree([(b'a', blob), (b'b/c', blob)])
        self.assertChangesEqual([], self.empty_tree, self.empty_tree)
        self.assertChangesEqual([], tree, tree)
        self.assertChangesEqual([TreeChange(CHANGE_UNCHANGED, (b'a', F, blob.id), (b'a', F, blob.id)), TreeChange(CHANGE_UNCHANGED, (b'b/c', F, blob.id), (b'b/c', F, blob.id))], tree, tree, want_unchanged=True)

    def test_tree_changes_add_delete(self):
        blob_a = make_object(Blob, data=b'a')
        blob_b = make_object(Blob, data=b'b')
        tree = self.commit_tree([(b'a', blob_a, 33188), (b'x/b', blob_b, 33261)])
        self.assertChangesEqual([TreeChange.add((b'a', 33188, blob_a.id)), TreeChange.add((b'x/b', 33261, blob_b.id))], self.empty_tree, tree)
        self.assertChangesEqual([TreeChange.delete((b'a', 33188, blob_a.id)), TreeChange.delete((b'x/b', 33261, blob_b.id))], tree, self.empty_tree)

    def test_tree_changes_modify_contents(self):
        blob_a1 = make_object(Blob, data=b'a1')
        blob_a2 = make_object(Blob, data=b'a2')
        tree1 = self.commit_tree([(b'a', blob_a1)])
        tree2 = self.commit_tree([(b'a', blob_a2)])
        self.assertChangesEqual([TreeChange(CHANGE_MODIFY, (b'a', F, blob_a1.id), (b'a', F, blob_a2.id))], tree1, tree2)

    def test_tree_changes_modify_mode(self):
        blob_a = make_object(Blob, data=b'a')
        tree1 = self.commit_tree([(b'a', blob_a, 33188)])
        tree2 = self.commit_tree([(b'a', blob_a, 33261)])
        self.assertChangesEqual([TreeChange(CHANGE_MODIFY, (b'a', 33188, blob_a.id), (b'a', 33261, blob_a.id))], tree1, tree2)

    def test_tree_changes_change_type(self):
        blob_a1 = make_object(Blob, data=b'a')
        blob_a2 = make_object(Blob, data=b'/foo/bar')
        tree1 = self.commit_tree([(b'a', blob_a1, 33188)])
        tree2 = self.commit_tree([(b'a', blob_a2, 40960)])
        self.assertChangesEqual([TreeChange.delete((b'a', 33188, blob_a1.id)), TreeChange.add((b'a', 40960, blob_a2.id))], tree1, tree2)

    def test_tree_changes_change_type_same(self):
        blob_a1 = make_object(Blob, data=b'a')
        blob_a2 = make_object(Blob, data=b'/foo/bar')
        tree1 = self.commit_tree([(b'a', blob_a1, 33188)])
        tree2 = self.commit_tree([(b'a', blob_a2, 40960)])
        self.assertChangesEqual([TreeChange(CHANGE_MODIFY, (b'a', 33188, blob_a1.id), (b'a', 40960, blob_a2.id))], tree1, tree2, change_type_same=True)

    def test_tree_changes_to_tree(self):
        blob_a = make_object(Blob, data=b'a')
        blob_x = make_object(Blob, data=b'x')
        tree1 = self.commit_tree([(b'a', blob_a)])
        tree2 = self.commit_tree([(b'a/x', blob_x)])
        self.assertChangesEqual([TreeChange.delete((b'a', F, blob_a.id)), TreeChange.add((b'a/x', F, blob_x.id))], tree1, tree2)

    def test_tree_changes_complex(self):
        blob_a_1 = make_object(Blob, data=b'a1_1')
        blob_bx1_1 = make_object(Blob, data=b'bx1_1')
        blob_bx2_1 = make_object(Blob, data=b'bx2_1')
        blob_by1_1 = make_object(Blob, data=b'by1_1')
        blob_by2_1 = make_object(Blob, data=b'by2_1')
        tree1 = self.commit_tree([(b'a', blob_a_1), (b'b/x/1', blob_bx1_1), (b'b/x/2', blob_bx2_1), (b'b/y/1', blob_by1_1), (b'b/y/2', blob_by2_1)])
        blob_a_2 = make_object(Blob, data=b'a1_2')
        blob_bx1_2 = blob_bx1_1
        blob_by_2 = make_object(Blob, data=b'by_2')
        blob_c_2 = make_object(Blob, data=b'c_2')
        tree2 = self.commit_tree([(b'a', blob_a_2), (b'b/x/1', blob_bx1_2), (b'b/y', blob_by_2), (b'c', blob_c_2)])
        self.assertChangesEqual([TreeChange(CHANGE_MODIFY, (b'a', F, blob_a_1.id), (b'a', F, blob_a_2.id)), TreeChange.delete((b'b/x/2', F, blob_bx2_1.id)), TreeChange.add((b'b/y', F, blob_by_2.id)), TreeChange.delete((b'b/y/1', F, blob_by1_1.id)), TreeChange.delete((b'b/y/2', F, blob_by2_1.id)), TreeChange.add((b'c', F, blob_c_2.id))], tree1, tree2)

    def test_tree_changes_name_order(self):
        blob = make_object(Blob, data=b'a')
        tree1 = self.commit_tree([(b'a', blob), (b'a.', blob), (b'a..', blob)])
        tree2 = self.commit_tree([(b'a/x', blob), (b'a./x', blob), (b'a..', blob)])
        self.assertChangesEqual([TreeChange.delete((b'a', F, blob.id)), TreeChange.add((b'a/x', F, blob.id)), TreeChange.delete((b'a.', F, blob.id)), TreeChange.add((b'a./x', F, blob.id))], tree1, tree2)

    def test_tree_changes_prune(self):
        blob_a1 = make_object(Blob, data=b'a1')
        blob_a2 = make_object(Blob, data=b'a2')
        blob_x = make_object(Blob, data=b'x')
        tree1 = self.commit_tree([(b'a', blob_a1), (b'b/x', blob_x)])
        tree2 = self.commit_tree([(b'a', blob_a2), (b'b/x', blob_x)])
        subtree = self.store[tree1[b'b'][1]]
        for entry in subtree.items():
            del self.store[entry.sha]
        del self.store[subtree.id]
        self.assertChangesEqual([TreeChange(CHANGE_MODIFY, (b'a', F, blob_a1.id), (b'a', F, blob_a2.id))], tree1, tree2)

    def test_tree_changes_rename_detector(self):
        blob_a1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob_a2 = make_object(Blob, data=b'a\nb\nc\ne\n')
        blob_b = make_object(Blob, data=b'b')
        tree1 = self.commit_tree([(b'a', blob_a1), (b'b', blob_b)])
        tree2 = self.commit_tree([(b'c', blob_a2), (b'b', blob_b)])
        detector = RenameDetector(self.store)
        self.assertChangesEqual([TreeChange.delete((b'a', F, blob_a1.id)), TreeChange.add((b'c', F, blob_a2.id))], tree1, tree2)
        self.assertChangesEqual([TreeChange.delete((b'a', F, blob_a1.id)), TreeChange(CHANGE_UNCHANGED, (b'b', F, blob_b.id), (b'b', F, blob_b.id)), TreeChange.add((b'c', F, blob_a2.id))], tree1, tree2, want_unchanged=True)
        self.assertChangesEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob_a1.id), (b'c', F, blob_a2.id))], tree1, tree2, rename_detector=detector)
        self.assertChangesEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob_a1.id), (b'c', F, blob_a2.id)), TreeChange(CHANGE_UNCHANGED, (b'b', F, blob_b.id), (b'b', F, blob_b.id))], tree1, tree2, rename_detector=detector, want_unchanged=True)

    def assertChangesForMergeEqual(self, expected, parent_trees, merge_tree, **kwargs):
        parent_tree_ids = [t.id for t in parent_trees]
        actual = list(tree_changes_for_merge(self.store, parent_tree_ids, merge_tree.id, **kwargs))
        self.assertEqual(expected, actual)
        parent_tree_ids.reverse()
        expected = [list(reversed(cs)) for cs in expected]
        actual = list(tree_changes_for_merge(self.store, parent_tree_ids, merge_tree.id, **kwargs))
        self.assertEqual(expected, actual)

    def test_tree_changes_for_merge_add_no_conflict(self):
        blob = make_object(Blob, data=b'blob')
        parent1 = self.commit_tree([])
        parent2 = merge = self.commit_tree([(b'a', blob)])
        self.assertChangesForMergeEqual([], [parent1, parent2], merge)
        self.assertChangesForMergeEqual([], [parent2, parent2], merge)

    def test_tree_changes_for_merge_add_modify_conflict(self):
        blob1 = make_object(Blob, data=b'1')
        blob2 = make_object(Blob, data=b'2')
        parent1 = self.commit_tree([])
        parent2 = self.commit_tree([(b'a', blob1)])
        merge = self.commit_tree([(b'a', blob2)])
        self.assertChangesForMergeEqual([[TreeChange.add((b'a', F, blob2.id)), TreeChange(CHANGE_MODIFY, (b'a', F, blob1.id), (b'a', F, blob2.id))]], [parent1, parent2], merge)

    def test_tree_changes_for_merge_modify_modify_conflict(self):
        blob1 = make_object(Blob, data=b'1')
        blob2 = make_object(Blob, data=b'2')
        blob3 = make_object(Blob, data=b'3')
        parent1 = self.commit_tree([(b'a', blob1)])
        parent2 = self.commit_tree([(b'a', blob2)])
        merge = self.commit_tree([(b'a', blob3)])
        self.assertChangesForMergeEqual([[TreeChange(CHANGE_MODIFY, (b'a', F, blob1.id), (b'a', F, blob3.id)), TreeChange(CHANGE_MODIFY, (b'a', F, blob2.id), (b'a', F, blob3.id))]], [parent1, parent2], merge)

    def test_tree_changes_for_merge_modify_no_conflict(self):
        blob1 = make_object(Blob, data=b'1')
        blob2 = make_object(Blob, data=b'2')
        parent1 = self.commit_tree([(b'a', blob1)])
        parent2 = merge = self.commit_tree([(b'a', blob2)])
        self.assertChangesForMergeEqual([], [parent1, parent2], merge)

    def test_tree_changes_for_merge_delete_delete_conflict(self):
        blob1 = make_object(Blob, data=b'1')
        blob2 = make_object(Blob, data=b'2')
        parent1 = self.commit_tree([(b'a', blob1)])
        parent2 = self.commit_tree([(b'a', blob2)])
        merge = self.commit_tree([])
        self.assertChangesForMergeEqual([[TreeChange.delete((b'a', F, blob1.id)), TreeChange.delete((b'a', F, blob2.id))]], [parent1, parent2], merge)

    def test_tree_changes_for_merge_delete_no_conflict(self):
        blob = make_object(Blob, data=b'blob')
        has = self.commit_tree([(b'a', blob)])
        doesnt_have = self.commit_tree([])
        self.assertChangesForMergeEqual([], [has, has], doesnt_have)
        self.assertChangesForMergeEqual([], [has, doesnt_have], doesnt_have)

    def test_tree_changes_for_merge_octopus_no_conflict(self):
        r = list(range(5))
        blobs = [make_object(Blob, data=bytes(i)) for i in r]
        parents = [self.commit_tree([(b'a', blobs[i])]) for i in r]
        for i in r:
            self.assertChangesForMergeEqual([], parents, parents[i])

    def test_tree_changes_for_merge_octopus_modify_conflict(self):
        r = list(range(5))
        parent_blobs = [make_object(Blob, data=bytes(i)) for i in r]
        merge_blob = make_object(Blob, data=b'merge')
        parents = [self.commit_tree([(b'a', parent_blobs[i])]) for i in r]
        merge = self.commit_tree([(b'a', merge_blob)])
        expected = [[TreeChange(CHANGE_MODIFY, (b'a', F, parent_blobs[i].id), (b'a', F, merge_blob.id)) for i in r]]
        self.assertChangesForMergeEqual(expected, parents, merge)

    def test_tree_changes_for_merge_octopus_delete(self):
        blob1 = make_object(Blob, data=b'1')
        blob2 = make_object(Blob, data=b'3')
        parent1 = self.commit_tree([(b'a', blob1)])
        parent2 = self.commit_tree([(b'a', blob2)])
        parent3 = merge = self.commit_tree([])
        self.assertChangesForMergeEqual([], [parent1, parent1, parent1], merge)
        self.assertChangesForMergeEqual([], [parent1, parent1, parent3], merge)
        self.assertChangesForMergeEqual([], [parent1, parent3, parent3], merge)
        self.assertChangesForMergeEqual([[TreeChange.delete((b'a', F, blob1.id)), TreeChange.delete((b'a', F, blob2.id)), None]], [parent1, parent2, parent3], merge)

    def test_tree_changes_for_merge_add_add_same_conflict(self):
        blob = make_object(Blob, data=b'a\nb\nc\nd\n')
        parent1 = self.commit_tree([(b'a', blob)])
        parent2 = self.commit_tree([])
        merge = self.commit_tree([(b'b', blob)])
        add = TreeChange.add((b'b', F, blob.id))
        self.assertChangesForMergeEqual([[add, add]], [parent1, parent2], merge)

    def test_tree_changes_for_merge_add_exact_rename_conflict(self):
        blob = make_object(Blob, data=b'a\nb\nc\nd\n')
        parent1 = self.commit_tree([(b'a', blob)])
        parent2 = self.commit_tree([])
        merge = self.commit_tree([(b'b', blob)])
        self.assertChangesForMergeEqual([[TreeChange(CHANGE_RENAME, (b'a', F, blob.id), (b'b', F, blob.id)), TreeChange.add((b'b', F, blob.id))]], [parent1, parent2], merge, rename_detector=self.detector)

    def test_tree_changes_for_merge_add_content_rename_conflict(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob2 = make_object(Blob, data=b'a\nb\nc\ne\n')
        parent1 = self.commit_tree([(b'a', blob1)])
        parent2 = self.commit_tree([])
        merge = self.commit_tree([(b'b', blob2)])
        self.assertChangesForMergeEqual([[TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'b', F, blob2.id)), TreeChange.add((b'b', F, blob2.id))]], [parent1, parent2], merge, rename_detector=self.detector)

    def test_tree_changes_for_merge_modify_rename_conflict(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob2 = make_object(Blob, data=b'a\nb\nc\ne\n')
        parent1 = self.commit_tree([(b'a', blob1)])
        parent2 = self.commit_tree([(b'b', blob1)])
        merge = self.commit_tree([(b'b', blob2)])
        self.assertChangesForMergeEqual([[TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'b', F, blob2.id)), TreeChange(CHANGE_MODIFY, (b'b', F, blob1.id), (b'b', F, blob2.id))]], [parent1, parent2], merge, rename_detector=self.detector)