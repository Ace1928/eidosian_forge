from io import StringIO
from .. import add, errors, tests
from ..bzr import inventory
class TestAddFrom(tests.TestCaseWithTransport):
    """Tests for AddFromBaseAction"""

    def make_base_tree(self):
        self.base_tree = self.make_branch_and_tree('base')
        self.build_tree(['base/a', 'base/b', 'base/dir/', 'base/dir/a', 'base/dir/subdir/', 'base/dir/subdir/b'])
        self.base_tree.add(['a', 'b', 'dir', 'dir/a', 'dir/subdir', 'dir/subdir/b'])
        self.base_tree.commit('creating initial tree.')

    def add_helper(self, base_tree, base_path, new_tree, file_list, should_print=False):
        to_file = StringIO()
        base_tree.lock_read()
        try:
            new_tree.lock_write()
            try:
                action = add.AddFromBaseAction(base_tree, base_path, to_file=to_file, should_print=should_print)
                new_tree.smart_add(file_list, action=action)
            finally:
                new_tree.unlock()
        finally:
            base_tree.unlock()
        return to_file.getvalue()

    def test_copy_all(self):
        self.make_base_tree()
        new_tree = self.make_branch_and_tree('new')
        files = ['a', 'b', 'dir/', 'dir/a', 'dir/subdir/', 'dir/subdir/b']
        self.build_tree(['new/' + fn for fn in files])
        self.add_helper(self.base_tree, '', new_tree, ['new'])
        for fn in files:
            base_file_id = self.base_tree.path2id(fn)
            new_file_id = new_tree.path2id(fn)
            self.assertEqual(base_file_id, new_file_id)

    def test_copy_from_dir(self):
        self.make_base_tree()
        new_tree = self.make_branch_and_tree('new')
        self.build_tree(['new/a', 'new/b', 'new/c', 'new/subdir/', 'new/subdir/b', 'new/subdir/d'])
        new_tree.set_root_id(self.base_tree.path2id(''))
        self.add_helper(self.base_tree, 'dir', new_tree, ['new'])
        self.assertEqual(self.base_tree.path2id('a'), new_tree.path2id('a'))
        self.assertEqual(self.base_tree.path2id('b'), new_tree.path2id('b'))
        self.assertEqual(self.base_tree.path2id('dir/subdir'), new_tree.path2id('subdir'))
        self.assertEqual(self.base_tree.path2id('dir/subdir/b'), new_tree.path2id('subdir/b'))
        c_id = new_tree.path2id('c')
        self.assertNotEqual(None, c_id)
        self.base_tree.lock_read()
        self.addCleanup(self.base_tree.unlock)
        self.assertRaises(errors.NoSuchId, self.base_tree.id2path, c_id)
        d_id = new_tree.path2id('subdir/d')
        self.assertNotEqual(None, d_id)
        self.assertRaises(errors.NoSuchId, self.base_tree.id2path, d_id)

    def test_copy_existing_dir(self):
        self.make_base_tree()
        new_tree = self.make_branch_and_tree('new')
        self.build_tree(['new/subby/', 'new/subby/a', 'new/subby/b'])
        subdir_file_id = self.base_tree.path2id('dir/subdir')
        new_tree.add(['subby'], ids=[subdir_file_id])
        self.add_helper(self.base_tree, '', new_tree, ['new'])
        self.assertEqual(self.base_tree.path2id('dir/subdir/b'), new_tree.path2id('subby/b'))
        a_id = new_tree.path2id('subby/a')
        self.assertNotEqual(None, a_id)
        self.base_tree.lock_read()
        self.addCleanup(self.base_tree.unlock)
        self.assertRaises(errors.NoSuchId, self.base_tree.id2path, a_id)