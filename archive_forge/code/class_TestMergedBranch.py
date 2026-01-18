import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
class TestMergedBranch(per_workingtree.TestCaseWithWorkingTree):

    def make_inner_branch(self):
        bld_inner = self.make_branch_builder('inner')
        bld_inner.start_series()
        rev1 = bld_inner.build_snapshot(None, [('add', ('', None, 'directory', '')), ('add', ('dir', None, 'directory', '')), ('add', ('dir/file1', None, 'file', b'file1 content\n')), ('add', ('file3', None, 'file', b'file3 content\n'))])
        rev4 = bld_inner.build_snapshot([rev1], [('add', ('file4', None, 'file', b'file4 content\n'))])
        rev5 = bld_inner.build_snapshot([rev4], [('rename', ('file4', 'dir/file4'))])
        rev3 = bld_inner.build_snapshot([rev1], [('modify', ('file3', b'new file3 contents\n'))])
        rev2 = bld_inner.build_snapshot([rev1], [('add', ('dir/file2', None, 'file', b'file2 content\n'))])
        bld_inner.finish_series()
        br = bld_inner.get_branch()
        return (br, [rev1, rev2, rev3, rev4, rev5])

    def assertTreeLayout(self, expected, tree):
        with tree.lock_read():
            actual = [e[0] for e in tree.list_files()]
            actual = sorted(actual)
            self.assertEqual(expected, actual)

    def make_outer_tree(self):
        outer = self.make_branch_and_tree('outer')
        self.build_tree_contents([('outer/foo', b'foo')])
        outer.add('foo')
        outer.commit('added foo')
        inner, revs = self.make_inner_branch()
        outer.merge_from_branch(inner, to_revision=revs[0], from_revision=b'null:')
        if outer.supports_setting_file_ids():
            outer.set_root_id(outer.basis_tree().path2id(''))
        outer.commit('merge inner branch')
        outer.mkdir('dir-outer')
        outer.move(['dir', 'file3'], to_dir='dir-outer')
        outer.commit('rename imported dir and file3 to dir-outer')
        return (outer, inner, revs)

    def test_file1_deleted_in_dir(self):
        outer, inner, revs = self.make_outer_tree()
        outer.remove(['dir-outer/dir/file1'], keep_files=False)
        outer.commit('delete file1')
        outer.merge_from_branch(inner)
        outer.commit('merge the rest')
        if outer.supports_rename_tracking():
            self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file2', 'dir-outer/file3', 'foo'], outer)
        else:
            self.assertTreeLayout(['dir', 'dir-outer', 'dir-outer/dir', 'dir-outer/file3', 'dir/file2', 'foo'], outer)

    def test_file3_deleted_in_root(self):
        outer, inner, revs = self.make_outer_tree()
        outer.remove(['dir-outer/file3'], keep_files=False)
        outer.commit('delete file3')
        outer.merge_from_branch(inner)
        outer.commit('merge the rest')
        if outer.supports_rename_tracking():
            self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'dir-outer/dir/file2', 'foo'], outer)
        else:
            self.assertTreeLayout(['dir', 'dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'dir/file2', 'foo'], outer)

    def test_file3_in_root_conflicted(self):
        outer, inner, revs = self.make_outer_tree()
        outer.remove(['dir-outer/file3'], keep_files=False)
        outer.commit('delete file3')
        nb_conflicts = outer.merge_from_branch(inner, to_revision=revs[2])
        if outer.supports_rename_tracking():
            self.assertEqual(4, len(nb_conflicts))
        else:
            self.assertEqual(1, len(nb_conflicts))
        self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'file3.BASE', 'file3.OTHER', 'foo'], outer)

    def test_file4_added_in_root(self):
        outer, inner, revs = self.make_outer_tree()
        nb_conflicts = outer.merge_from_branch(inner, to_revision=revs[3])
        if outer.supports_rename_tracking():
            self.assertEqual(1, len(nb_conflicts))
        else:
            self.assertEqual(0, len(nb_conflicts))
        self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'dir-outer/file3', 'file4', 'foo'], outer)

    def test_file4_added_then_renamed(self):
        outer, inner, revs = self.make_outer_tree()
        nb_conflicts = outer.merge_from_branch(inner, to_revision=revs[3])
        if outer.supports_rename_tracking():
            self.assertEqual(1, len(nb_conflicts))
        else:
            self.assertEqual(0, len(nb_conflicts))
        try:
            outer.set_conflicts([])
        except errors.UnsupportedOperation:
            pass
        outer.commit('added file4')
        nb_conflicts = outer.merge_from_branch(inner, to_revision=revs[4])
        if outer.supports_rename_tracking():
            self.assertEqual(1, len(nb_conflicts))
            self.assertTreeLayout(['dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'dir-outer/dir/file4', 'dir-outer/file3', 'foo'], outer)
        else:
            if outer.has_versioned_directories():
                self.assertEqual(2, len(nb_conflicts))
            else:
                self.assertEqual(0, len(nb_conflicts))
            self.assertTreeLayout(['dir', 'dir-outer', 'dir-outer/dir', 'dir-outer/dir/file1', 'dir-outer/file3', 'dir/file4', 'foo'], outer)