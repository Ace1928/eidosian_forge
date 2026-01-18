import os
import sys
import tempfile
import breezy
from .. import controldir, errors, merge_directive, osutils
from ..bzr import generate_ids
from ..bzr.conflicts import ContentsConflict, PathConflict, TextConflict
from ..merge import Diff3Merger, Merge3Merger, Merger, WeaveMerger
from ..osutils import getcwd, pathjoin
from ..workingtree import WorkingTree
from . import TestCaseWithTransport, TestSkipped, features
class MergeTest(TestCaseWithTransport):

    def test_change_name(self):
        """Test renames"""
        builder = MergeBuilder(getcwd())
        name1 = builder.add_file(builder.root(), 'name1', b'hello1', True, file_id=b'1')
        builder.change_name(name1, other='name2')
        name3 = builder.add_file(builder.root(), 'name3', b'hello2', True, file_id=b'2')
        builder.change_name(name3, base='name4')
        name5 = builder.add_file(builder.root(), 'name5', b'hello3', True, file_id=b'3')
        builder.change_name(name5, this='name6')
        builder.merge()
        builder.cleanup()
        builder = MergeBuilder(getcwd())
        name1 = builder.add_file(builder.root(), 'name1', b'hello1', False, file_id=b'1')
        builder.change_name(name1, other='name2', this='name3')
        conflicts = builder.merge()
        self.assertEqual(conflicts, [PathConflict('name3', 'name2', b'1')])
        builder.cleanup()

    def test_merge_one(self):
        builder = MergeBuilder(getcwd())
        name1 = builder.add_file(builder.root(), 'name1', b'hello1', True, file_id=b'1')
        builder.change_contents(name1, other=b'text4')
        name2 = builder.add_file(builder.root(), 'name2', b'hello1', True, file_id=b'2')
        builder.change_contents(name2, other=b'text4')
        builder.merge(interesting_files=['name1'])
        self.assertEqual(builder.this.get_file('name1').read(), b'text4')
        self.assertEqual(builder.this.get_file('name2').read(), b'hello1')
        builder.cleanup()

    def test_file_moves(self):
        """Test moves"""
        builder = MergeBuilder(getcwd())
        dir1 = builder.add_dir(builder.root(), 'dir1', file_id=b'1')
        dir2 = builder.add_dir(builder.root(), 'dir2', file_id=b'2')
        file1 = builder.add_file(dir1, 'file1', b'hello1', True, file_id=b'3')
        file2 = builder.add_file(dir1, 'file2', b'hello2', True, file_id=b'4')
        file3 = builder.add_file(dir1, 'file3', b'hello3', True, file_id=b'5')
        builder.change_parent(file1, other=b'2')
        builder.change_parent(file2, this=b'2')
        builder.change_parent(file3, base=b'2')
        builder.merge()
        builder.cleanup()
        builder = MergeBuilder(getcwd())
        dir1 = builder.add_dir(builder.root(), 'dir1', file_id=b'1')
        builder.add_dir(builder.root(), 'dir2', file_id=b'2')
        builder.add_dir(builder.root(), 'dir3', file_id=b'3')
        file1 = builder.add_file(dir1, 'file1', b'hello1', False, file_id=b'4')
        builder.change_parent(file1, other=b'2', this=b'3')
        conflicts = builder.merge()
        path2 = pathjoin('dir2', 'file1')
        path3 = pathjoin('dir3', 'file1')
        self.assertEqual(conflicts, [PathConflict(path3, path2, b'4')])
        builder.cleanup()

    def test_contents_merge(self):
        """Test merge3 merging"""
        self.do_contents_test(Merge3Merger)

    def test_contents_merge2(self):
        """Test diff3 merging"""
        if sys.platform == 'win32':
            raise TestSkipped('diff3 does not have --binary flag and therefore always fails on win32')
        try:
            self.do_contents_test(Diff3Merger)
        except errors.NoDiff3:
            raise TestSkipped('diff3 not available')

    def test_contents_merge3(self):
        """Test diff3 merging"""
        self.do_contents_test(WeaveMerger)

    def test_reprocess_weave(self):
        builder = MergeBuilder(getcwd())
        blah = builder.add_file(builder.root(), 'blah', b'a', False, file_id=b'a')
        builder.change_contents(blah, this=b'b\nc\nd\ne\n', other=b'z\nc\nd\ny\n')
        builder.merge(WeaveMerger, reprocess=True)
        expected = b'<<<<<<< TREE\nb\n=======\nz\n>>>>>>> MERGE-SOURCE\nc\nd\n<<<<<<< TREE\ne\n=======\ny\n>>>>>>> MERGE-SOURCE\n'
        self.assertEqualDiff(builder.this.get_file_text('blah'), expected)
        builder.cleanup()

    def do_contents_test(self, merge_factory):
        """Test merging with specified ContentsChange factory"""
        builder = self.contents_test_success(merge_factory)
        builder.cleanup()
        self.contents_test_conflicts(merge_factory)

    def contents_test_success(self, merge_factory):
        builder = MergeBuilder(getcwd())
        name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
        builder.change_contents(name1, other=b'text4')
        name3 = builder.add_file(builder.root(), 'name3', b'text2', False, file_id=b'2')
        builder.change_contents(name3, base=b'text5')
        builder.add_file(builder.root(), 'name5', b'text3', True, file_id=b'3')
        name6 = builder.add_file(builder.root(), 'name6', b'text4', True, file_id=b'4')
        builder.remove_file(name6, base=True)
        name7 = builder.add_file(builder.root(), 'name7', b'a\nb\nc\nd\ne\nf\n', True, file_id=b'5')
        builder.change_contents(name7, other=b'a\nz\nc\nd\ne\nf\n', this=b'a\nb\nc\nd\ne\nz\n')
        conflicts = builder.merge(merge_factory)
        try:
            self.assertEqual([], conflicts)
            self.assertEqual(b'text4', builder.this.get_file('name1').read())
            self.assertEqual(b'text2', builder.this.get_file('name3').read())
            self.assertEqual(b'a\nz\nc\nd\ne\nz\n', builder.this.get_file('name7').read())
            self.assertTrue(builder.this.is_executable('name1'))
            self.assertFalse(builder.this.is_executable('name3'))
            self.assertTrue(builder.this.is_executable('name5'))
        except:
            builder.unlock()
            raise
        return builder

    def contents_test_conflicts(self, merge_factory):
        builder = MergeBuilder(getcwd())
        name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
        builder.change_contents(name1, other=b'text4', this=b'text3')
        name2 = builder.add_file(builder.root(), 'name2', b'text1', True, file_id=b'2')
        builder.change_contents(name2, other=b'\x00', this=b'text3')
        name3 = builder.add_file(builder.root(), 'name3', b'text5', False, file_id=b'3')
        builder.change_perms(name3, this=True)
        builder.change_contents(name3, this=b'moretext')
        builder.remove_file(name3, other=True)
        conflicts = builder.merge(merge_factory)
        self.assertEqual(conflicts, [TextConflict('name1', file_id=b'1'), ContentsConflict('name2', file_id=b'2'), ContentsConflict('name3', file_id=b'3')])
        with builder.this.get_file(builder.this.id2path(b'2')) as f:
            self.assertEqual(f.read(), b'\x00')
        builder.cleanup()

    def test_symlink_conflicts(self):
        if sys.platform != 'win32':
            builder = MergeBuilder(getcwd())
            name2 = builder.add_symlink(builder.root(), 'name2', 'target1', file_id=b'2')
            builder.change_target(name2, other='target4', base='text3')
            conflicts = builder.merge()
            self.assertEqual(conflicts, [ContentsConflict('name2', file_id=b'2')])
            builder.cleanup()

    def test_symlink_merge(self):
        if sys.platform != 'win32':
            builder = MergeBuilder(getcwd())
            name1 = builder.add_symlink(builder.root(), 'name1', 'target1', file_id=b'1')
            name2 = builder.add_symlink(builder.root(), 'name2', 'target1', file_id=b'2')
            name3 = builder.add_symlink(builder.root(), 'name3', 'target1', file_id=b'3')
            builder.change_target(name1, this=b'target2')
            builder.change_target(name2, base=b'target2')
            builder.change_target(name3, other=b'target2')
            builder.merge()
            self.assertEqual(builder.this.get_symlink_target('name1'), 'target2')
            self.assertEqual(builder.this.get_symlink_target('name2'), 'target1')
            self.assertEqual(builder.this.get_symlink_target('name3'), 'target2')
            builder.cleanup()

    def test_no_passive_add(self):
        builder = MergeBuilder(getcwd())
        name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
        builder.remove_file(name1, this=True)
        builder.merge()
        builder.cleanup()

    def test_perms_merge(self):
        builder = MergeBuilder(getcwd())
        name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
        builder.change_perms(name1, other=False)
        name2 = builder.add_file(builder.root(), 'name2', b'text2', True, file_id=b'2')
        builder.change_perms(name2, base=False)
        name3 = builder.add_file(builder.root(), 'name3', b'text3', True, file_id=b'3')
        builder.change_perms(name3, this=False)
        name4 = builder.add_file(builder.root(), 'name4', b'text4', False, file_id=b'4')
        builder.change_perms(name4, this=True)
        builder.remove_file(name4, base=True)
        builder.merge()
        self.assertIs(builder.this.is_executable('name1'), False)
        self.assertIs(builder.this.is_executable('name2'), True)
        self.assertIs(builder.this.is_executable('name3'), False)
        builder.cleanup()

    def test_new_suffix(self):
        builder = MergeBuilder(getcwd())
        name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
        builder.change_contents(name1, other=b'text3')
        builder.add_file(builder.root(), 'name1.new', b'text2', True, file_id=b'2')
        builder.merge()
        os.lstat(builder.this.abspath('name1.new'))
        builder.cleanup()

    def test_spurious_conflict(self):
        builder = MergeBuilder(getcwd())
        name1 = builder.add_file(builder.root(), 'name1', b'text1', False, file_id=b'1')
        builder.remove_file(name1, other=True)
        builder.add_file(builder.root(), 'name1', b'text1', False, this=False, base=False, file_id=b'2')
        conflicts = builder.merge()
        self.assertEqual(conflicts, [])
        builder.cleanup()

    def test_merge_one_renamed(self):
        builder = MergeBuilder(getcwd())
        name1 = builder.add_file(builder.root(), 'name1', b'text1a', False, file_id=b'1')
        builder.change_name(name1, this='name2')
        builder.change_contents(name1, other=b'text2')
        builder.merge(interesting_files=['name2'])
        self.assertEqual(b'text2', builder.this.get_file('name2').read())
        builder.cleanup()