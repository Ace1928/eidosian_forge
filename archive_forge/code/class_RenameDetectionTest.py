from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
class RenameDetectionTest(DiffTestCase):

    def _do_test_count_blocks(self, count_blocks):
        blob = make_object(Blob, data=b'a\nb\na\n')
        self.assertBlockCountEqual({b'a\n': 4, b'b\n': 2}, count_blocks(blob))
    test_count_blocks = functest_builder(_do_test_count_blocks, _count_blocks_py)
    test_count_blocks_extension = ext_functest_builder(_do_test_count_blocks, _count_blocks)

    def _do_test_count_blocks_no_newline(self, count_blocks):
        blob = make_object(Blob, data=b'a\na')
        self.assertBlockCountEqual({b'a\n': 2, b'a': 1}, _count_blocks(blob))
    test_count_blocks_no_newline = functest_builder(_do_test_count_blocks_no_newline, _count_blocks_py)
    test_count_blocks_no_newline_extension = ext_functest_builder(_do_test_count_blocks_no_newline, _count_blocks)

    def assertBlockCountEqual(self, expected, got):
        self.assertEqual({hash(l) & 4294967295: c for l, c in expected.items()}, {h & 4294967295: c for h, c in got.items()})

    def _do_test_count_blocks_chunks(self, count_blocks):
        blob = ShaFile.from_raw_chunks(Blob.type_num, [b'a\nb', b'\na\n'])
        self.assertBlockCountEqual({b'a\n': 4, b'b\n': 2}, _count_blocks(blob))
    test_count_blocks_chunks = functest_builder(_do_test_count_blocks_chunks, _count_blocks_py)
    test_count_blocks_chunks_extension = ext_functest_builder(_do_test_count_blocks_chunks, _count_blocks)

    def _do_test_count_blocks_long_lines(self, count_blocks):
        a = b'a' * 64
        data = a + b'xxx\ny\n' + a + b'zzz\n'
        blob = make_object(Blob, data=data)
        self.assertBlockCountEqual({b'a' * 64: 128, b'xxx\n': 4, b'y\n': 2, b'zzz\n': 4}, _count_blocks(blob))
    test_count_blocks_long_lines = functest_builder(_do_test_count_blocks_long_lines, _count_blocks_py)
    test_count_blocks_long_lines_extension = ext_functest_builder(_do_test_count_blocks_long_lines, _count_blocks)

    def assertSimilar(self, expected_score, blob1, blob2):
        self.assertEqual(expected_score, _similarity_score(blob1, blob2))
        self.assertEqual(expected_score, _similarity_score(blob2, blob1))

    def test_similarity_score(self):
        blob0 = make_object(Blob, data=b'')
        blob1 = make_object(Blob, data=b'ab\ncd\ncd\n')
        blob2 = make_object(Blob, data=b'ab\n')
        blob3 = make_object(Blob, data=b'cd\n')
        blob4 = make_object(Blob, data=b'cd\ncd\n')
        self.assertSimilar(100, blob0, blob0)
        self.assertSimilar(0, blob0, blob1)
        self.assertSimilar(33, blob1, blob2)
        self.assertSimilar(33, blob1, blob3)
        self.assertSimilar(66, blob1, blob4)
        self.assertSimilar(0, blob2, blob3)
        self.assertSimilar(50, blob3, blob4)

    def test_similarity_score_cache(self):
        blob1 = make_object(Blob, data=b'ab\ncd\n')
        blob2 = make_object(Blob, data=b'ab\n')
        block_cache = {}
        self.assertEqual(50, _similarity_score(blob1, blob2, block_cache=block_cache))
        self.assertEqual({blob1.id, blob2.id}, set(block_cache))

        def fail_chunks():
            self.fail('Unexpected call to as_raw_chunks()')
        blob1.as_raw_chunks = blob2.as_raw_chunks = fail_chunks
        blob1.raw_length = lambda: 6
        blob2.raw_length = lambda: 3
        self.assertEqual(50, _similarity_score(blob1, blob2, block_cache=block_cache))

    def test_tree_entry_sort(self):
        sha = 'abcd' * 10
        expected_entries = [TreeChange.add(TreeEntry(b'aaa', F, sha)), TreeChange(CHANGE_COPY, TreeEntry(b'bbb', F, sha), TreeEntry(b'aab', F, sha)), TreeChange(CHANGE_MODIFY, TreeEntry(b'bbb', F, sha), TreeEntry(b'bbb', F, b'dabc' * 10)), TreeChange(CHANGE_RENAME, TreeEntry(b'bbc', F, sha), TreeEntry(b'ddd', F, sha)), TreeChange.delete(TreeEntry(b'ccc', F, sha))]
        for perm in permutations(expected_entries):
            self.assertEqual(expected_entries, sorted(perm, key=_tree_change_key))

    def detect_renames(self, tree1, tree2, want_unchanged=False, **kwargs):
        detector = RenameDetector(self.store, **kwargs)
        return detector.changes_with_renames(tree1.id, tree2.id, want_unchanged=want_unchanged)

    def test_no_renames(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob2 = make_object(Blob, data=b'a\nb\ne\nf\n')
        blob3 = make_object(Blob, data=b'a\nb\ng\nh\n')
        tree1 = self.commit_tree([(b'a', blob1), (b'b', blob2)])
        tree2 = self.commit_tree([(b'a', blob1), (b'b', blob3)])
        self.assertEqual([TreeChange(CHANGE_MODIFY, (b'b', F, blob2.id), (b'b', F, blob3.id))], self.detect_renames(tree1, tree2))

    def test_exact_rename_one_to_one(self):
        blob1 = make_object(Blob, data=b'1')
        blob2 = make_object(Blob, data=b'2')
        tree1 = self.commit_tree([(b'a', blob1), (b'b', blob2)])
        tree2 = self.commit_tree([(b'c', blob1), (b'd', blob2)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'c', F, blob1.id)), TreeChange(CHANGE_RENAME, (b'b', F, blob2.id), (b'd', F, blob2.id))], self.detect_renames(tree1, tree2))

    def test_exact_rename_split_different_type(self):
        blob = make_object(Blob, data=b'/foo')
        tree1 = self.commit_tree([(b'a', blob, 33188)])
        tree2 = self.commit_tree([(b'a', blob, 40960)])
        self.assertEqual([TreeChange.add((b'a', 40960, blob.id)), TreeChange.delete((b'a', 33188, blob.id))], self.detect_renames(tree1, tree2))

    def test_exact_rename_and_different_type(self):
        blob1 = make_object(Blob, data=b'1')
        blob2 = make_object(Blob, data=b'2')
        tree1 = self.commit_tree([(b'a', blob1)])
        tree2 = self.commit_tree([(b'a', blob2, 40960), (b'b', blob1)])
        self.assertEqual([TreeChange.add((b'a', 40960, blob2.id)), TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'b', F, blob1.id))], self.detect_renames(tree1, tree2))

    def test_exact_rename_one_to_many(self):
        blob = make_object(Blob, data=b'1')
        tree1 = self.commit_tree([(b'a', blob)])
        tree2 = self.commit_tree([(b'b', blob), (b'c', blob)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob.id), (b'b', F, blob.id)), TreeChange(CHANGE_COPY, (b'a', F, blob.id), (b'c', F, blob.id))], self.detect_renames(tree1, tree2))

    def test_exact_rename_many_to_one(self):
        blob = make_object(Blob, data=b'1')
        tree1 = self.commit_tree([(b'a', blob), (b'b', blob)])
        tree2 = self.commit_tree([(b'c', blob)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob.id), (b'c', F, blob.id)), TreeChange.delete((b'b', F, blob.id))], self.detect_renames(tree1, tree2))

    def test_exact_rename_many_to_many(self):
        blob = make_object(Blob, data=b'1')
        tree1 = self.commit_tree([(b'a', blob), (b'b', blob)])
        tree2 = self.commit_tree([(b'c', blob), (b'd', blob), (b'e', blob)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob.id), (b'c', F, blob.id)), TreeChange(CHANGE_COPY, (b'a', F, blob.id), (b'e', F, blob.id)), TreeChange(CHANGE_RENAME, (b'b', F, blob.id), (b'd', F, blob.id))], self.detect_renames(tree1, tree2))

    def test_exact_copy_modify(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob2 = make_object(Blob, data=b'a\nb\nc\ne\n')
        tree1 = self.commit_tree([(b'a', blob1)])
        tree2 = self.commit_tree([(b'a', blob2), (b'b', blob1)])
        self.assertEqual([TreeChange(CHANGE_MODIFY, (b'a', F, blob1.id), (b'a', F, blob2.id)), TreeChange(CHANGE_COPY, (b'a', F, blob1.id), (b'b', F, blob1.id))], self.detect_renames(tree1, tree2))

    def test_exact_copy_change_mode(self):
        blob = make_object(Blob, data=b'a\nb\nc\nd\n')
        tree1 = self.commit_tree([(b'a', blob)])
        tree2 = self.commit_tree([(b'a', blob, 33261), (b'b', blob)])
        self.assertEqual([TreeChange(CHANGE_MODIFY, (b'a', F, blob.id), (b'a', 33261, blob.id)), TreeChange(CHANGE_COPY, (b'a', F, blob.id), (b'b', F, blob.id))], self.detect_renames(tree1, tree2))

    def test_rename_threshold(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\n')
        blob2 = make_object(Blob, data=b'a\nb\nd\n')
        tree1 = self.commit_tree([(b'a', blob1)])
        tree2 = self.commit_tree([(b'b', blob2)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'b', F, blob2.id))], self.detect_renames(tree1, tree2, rename_threshold=50))
        self.assertEqual([TreeChange.delete((b'a', F, blob1.id)), TreeChange.add((b'b', F, blob2.id))], self.detect_renames(tree1, tree2, rename_threshold=75))

    def test_content_rename_max_files(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\nd')
        blob4 = make_object(Blob, data=b'a\nb\nc\ne\n')
        blob2 = make_object(Blob, data=b'e\nf\ng\nh\n')
        blob3 = make_object(Blob, data=b'e\nf\ng\ni\n')
        tree1 = self.commit_tree([(b'a', blob1), (b'b', blob2)])
        tree2 = self.commit_tree([(b'c', blob3), (b'd', blob4)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'd', F, blob4.id)), TreeChange(CHANGE_RENAME, (b'b', F, blob2.id), (b'c', F, blob3.id))], self.detect_renames(tree1, tree2))
        self.assertEqual([TreeChange.delete((b'a', F, blob1.id)), TreeChange.delete((b'b', F, blob2.id)), TreeChange.add((b'c', F, blob3.id)), TreeChange.add((b'd', F, blob4.id))], self.detect_renames(tree1, tree2, max_files=1))

    def test_content_rename_one_to_one(self):
        b11 = make_object(Blob, data=b'a\nb\nc\nd\n')
        b12 = make_object(Blob, data=b'a\nb\nc\ne\n')
        b21 = make_object(Blob, data=b'e\nf\ng\n\nh')
        b22 = make_object(Blob, data=b'e\nf\ng\n\ni')
        tree1 = self.commit_tree([(b'a', b11), (b'b', b21)])
        tree2 = self.commit_tree([(b'c', b12), (b'd', b22)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, b11.id), (b'c', F, b12.id)), TreeChange(CHANGE_RENAME, (b'b', F, b21.id), (b'd', F, b22.id))], self.detect_renames(tree1, tree2))

    def test_content_rename_one_to_one_ordering(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\nd\ne\nf\n')
        blob2 = make_object(Blob, data=b'a\nb\nc\nd\ng\nh\n')
        blob3 = make_object(Blob, data=b'a\nb\nc\nd\ng\ni\n')
        tree1 = self.commit_tree([(b'a', blob1), (b'b', blob2)])
        tree2 = self.commit_tree([(b'c', blob3)])
        self.assertEqual([TreeChange.delete((b'a', F, blob1.id)), TreeChange(CHANGE_RENAME, (b'b', F, blob2.id), (b'c', F, blob3.id))], self.detect_renames(tree1, tree2))
        tree3 = self.commit_tree([(b'a', blob2), (b'b', blob1)])
        tree4 = self.commit_tree([(b'c', blob3)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob2.id), (b'c', F, blob3.id)), TreeChange.delete((b'b', F, blob1.id))], self.detect_renames(tree3, tree4))

    def test_content_rename_one_to_many(self):
        blob1 = make_object(Blob, data=b'aa\nb\nc\nd\ne\n')
        blob2 = make_object(Blob, data=b'ab\nb\nc\nd\ne\n')
        blob3 = make_object(Blob, data=b'aa\nb\nc\nd\nf\n')
        tree1 = self.commit_tree([(b'a', blob1)])
        tree2 = self.commit_tree([(b'b', blob2), (b'c', blob3)])
        self.assertEqual([TreeChange(CHANGE_COPY, (b'a', F, blob1.id), (b'b', F, blob2.id)), TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'c', F, blob3.id))], self.detect_renames(tree1, tree2))

    def test_content_rename_many_to_one(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob2 = make_object(Blob, data=b'a\nb\nc\ne\n')
        blob3 = make_object(Blob, data=b'a\nb\nc\nf\n')
        tree1 = self.commit_tree([(b'a', blob1), (b'b', blob2)])
        tree2 = self.commit_tree([(b'c', blob3)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'c', F, blob3.id)), TreeChange.delete((b'b', F, blob2.id))], self.detect_renames(tree1, tree2))

    def test_content_rename_many_to_many(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob2 = make_object(Blob, data=b'a\nb\nc\ne\n')
        blob3 = make_object(Blob, data=b'a\nb\nc\nf\n')
        blob4 = make_object(Blob, data=b'a\nb\nc\ng\n')
        tree1 = self.commit_tree([(b'a', blob1), (b'b', blob2)])
        tree2 = self.commit_tree([(b'c', blob3), (b'd', blob4)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'c', F, blob3.id)), TreeChange(CHANGE_COPY, (b'a', F, blob1.id), (b'd', F, blob4.id)), TreeChange.delete((b'b', F, blob2.id))], self.detect_renames(tree1, tree2))

    def test_content_rename_with_more_deletions(self):
        blob1 = make_object(Blob, data=b'')
        tree1 = self.commit_tree([(b'a', blob1), (b'b', blob1), (b'c', blob1), (b'd', blob1)])
        tree2 = self.commit_tree([(b'e', blob1), (b'f', blob1), (b'g', blob1)])
        self.maxDiff = None
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'e', F, blob1.id)), TreeChange(CHANGE_RENAME, (b'b', F, blob1.id), (b'f', F, blob1.id)), TreeChange(CHANGE_RENAME, (b'c', F, blob1.id), (b'g', F, blob1.id)), TreeChange.delete((b'd', F, blob1.id))], self.detect_renames(tree1, tree2))

    def test_content_rename_gitlink(self):
        blob1 = make_object(Blob, data=b'blob1')
        blob2 = make_object(Blob, data=b'blob2')
        link1 = b'1' * 40
        link2 = b'2' * 40
        tree1 = self.commit_tree([(b'a', blob1), (b'b', link1, 57344)])
        tree2 = self.commit_tree([(b'c', blob2), (b'd', link2, 57344)])
        self.assertEqual([TreeChange.delete((b'a', 33188, blob1.id)), TreeChange.delete((b'b', 57344, link1)), TreeChange.add((b'c', 33188, blob2.id)), TreeChange.add((b'd', 57344, link2))], self.detect_renames(tree1, tree2))

    def test_exact_rename_swap(self):
        blob1 = make_object(Blob, data=b'1')
        blob2 = make_object(Blob, data=b'2')
        tree1 = self.commit_tree([(b'a', blob1), (b'b', blob2)])
        tree2 = self.commit_tree([(b'a', blob2), (b'b', blob1)])
        self.assertEqual([TreeChange(CHANGE_MODIFY, (b'a', F, blob1.id), (b'a', F, blob2.id)), TreeChange(CHANGE_MODIFY, (b'b', F, blob2.id), (b'b', F, blob1.id))], self.detect_renames(tree1, tree2))
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'b', F, blob1.id)), TreeChange(CHANGE_RENAME, (b'b', F, blob2.id), (b'a', F, blob2.id))], self.detect_renames(tree1, tree2, rewrite_threshold=50))

    def test_content_rename_swap(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob2 = make_object(Blob, data=b'e\nf\ng\nh\n')
        blob3 = make_object(Blob, data=b'a\nb\nc\ne\n')
        blob4 = make_object(Blob, data=b'e\nf\ng\ni\n')
        tree1 = self.commit_tree([(b'a', blob1), (b'b', blob2)])
        tree2 = self.commit_tree([(b'a', blob4), (b'b', blob3)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'b', F, blob3.id)), TreeChange(CHANGE_RENAME, (b'b', F, blob2.id), (b'a', F, blob4.id))], self.detect_renames(tree1, tree2, rewrite_threshold=60))

    def test_rewrite_threshold(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob2 = make_object(Blob, data=b'a\nb\nc\ne\n')
        blob3 = make_object(Blob, data=b'a\nb\nf\ng\n')
        tree1 = self.commit_tree([(b'a', blob1)])
        tree2 = self.commit_tree([(b'a', blob3), (b'b', blob2)])
        no_renames = [TreeChange(CHANGE_MODIFY, (b'a', F, blob1.id), (b'a', F, blob3.id)), TreeChange(CHANGE_COPY, (b'a', F, blob1.id), (b'b', F, blob2.id))]
        self.assertEqual(no_renames, self.detect_renames(tree1, tree2))
        self.assertEqual(no_renames, self.detect_renames(tree1, tree2, rewrite_threshold=40))
        self.assertEqual([TreeChange.add((b'a', F, blob3.id)), TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'b', F, blob2.id))], self.detect_renames(tree1, tree2, rewrite_threshold=80))

    def test_find_copies_harder_exact(self):
        blob = make_object(Blob, data=b'blob')
        tree1 = self.commit_tree([(b'a', blob)])
        tree2 = self.commit_tree([(b'a', blob), (b'b', blob)])
        self.assertEqual([TreeChange.add((b'b', F, blob.id))], self.detect_renames(tree1, tree2))
        self.assertEqual([TreeChange(CHANGE_COPY, (b'a', F, blob.id), (b'b', F, blob.id))], self.detect_renames(tree1, tree2, find_copies_harder=True))

    def test_find_copies_harder_content(self):
        blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob2 = make_object(Blob, data=b'a\nb\nc\ne\n')
        tree1 = self.commit_tree([(b'a', blob1)])
        tree2 = self.commit_tree([(b'a', blob1), (b'b', blob2)])
        self.assertEqual([TreeChange.add((b'b', F, blob2.id))], self.detect_renames(tree1, tree2))
        self.assertEqual([TreeChange(CHANGE_COPY, (b'a', F, blob1.id), (b'b', F, blob2.id))], self.detect_renames(tree1, tree2, find_copies_harder=True))

    def test_find_copies_harder_with_rewrites(self):
        blob_a1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob_a2 = make_object(Blob, data=b'f\ng\nh\ni\n')
        blob_b2 = make_object(Blob, data=b'a\nb\nc\ne\n')
        tree1 = self.commit_tree([(b'a', blob_a1)])
        tree2 = self.commit_tree([(b'a', blob_a2), (b'b', blob_b2)])
        self.assertEqual([TreeChange(CHANGE_MODIFY, (b'a', F, blob_a1.id), (b'a', F, blob_a2.id)), TreeChange(CHANGE_COPY, (b'a', F, blob_a1.id), (b'b', F, blob_b2.id))], self.detect_renames(tree1, tree2, find_copies_harder=True))
        self.assertEqual([TreeChange.add((b'a', F, blob_a2.id)), TreeChange(CHANGE_RENAME, (b'a', F, blob_a1.id), (b'b', F, blob_b2.id))], self.detect_renames(tree1, tree2, rewrite_threshold=50, find_copies_harder=True))

    def test_reuse_detector(self):
        blob = make_object(Blob, data=b'blob')
        tree1 = self.commit_tree([(b'a', blob)])
        tree2 = self.commit_tree([(b'b', blob)])
        detector = RenameDetector(self.store)
        changes = [TreeChange(CHANGE_RENAME, (b'a', F, blob.id), (b'b', F, blob.id))]
        self.assertEqual(changes, detector.changes_with_renames(tree1.id, tree2.id))
        self.assertEqual(changes, detector.changes_with_renames(tree1.id, tree2.id))

    def test_want_unchanged(self):
        blob_a1 = make_object(Blob, data=b'a\nb\nc\nd\n')
        blob_b = make_object(Blob, data=b'b')
        blob_c2 = make_object(Blob, data=b'a\nb\nc\ne\n')
        tree1 = self.commit_tree([(b'a', blob_a1), (b'b', blob_b)])
        tree2 = self.commit_tree([(b'c', blob_c2), (b'b', blob_b)])
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob_a1.id), (b'c', F, blob_c2.id))], self.detect_renames(tree1, tree2))
        self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob_a1.id), (b'c', F, blob_c2.id)), TreeChange(CHANGE_UNCHANGED, (b'b', F, blob_b.id), (b'b', F, blob_b.id))], self.detect_renames(tree1, tree2, want_unchanged=True))