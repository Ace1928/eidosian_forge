import contextlib
from breezy import errors, tests, transform, transport
from breezy.bzr.workingtree_4 import (DirStateRevisionTree, WorkingTreeFormat4,
from breezy.git.tree import GitRevisionTree
from breezy.git.workingtree import GitWorkingTreeFormat
from breezy.revisiontree import RevisionTree
from breezy.tests import features
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.tests.per_workingtree import make_scenario as wt_make_scenario
from breezy.tests.per_workingtree import make_scenarios as wt_make_scenarios
from breezy.workingtree import format_registry
class TestCaseWithTree(TestCaseWithControlDir):

    def make_branch_and_tree(self, relpath):
        bzrdir_format = self.workingtree_format.get_controldir_for_branch()
        made_control = self.make_controldir(relpath, format=bzrdir_format)
        made_control.create_repository()
        b = made_control.create_branch()
        if getattr(self, 'repo_is_remote', False):
            t = transport.get_transport(relpath)
            t.ensure_base()
            wt_dir = bzrdir_format.initialize_on_transport(t)
            branch_ref = wt_dir.set_branch_reference(b)
            wt = wt_dir.create_workingtree(None, from_branch=branch_ref)
        else:
            wt = self.workingtree_format.initialize(made_control)
        return wt

    def workingtree_to_test_tree(self, tree):
        return self._workingtree_to_test_tree(self, tree)

    def _convert_tree(self, tree, converter=None):
        """helper to convert using the converter or a supplied one."""
        if converter is None:
            converter = self.workingtree_to_test_tree
        return converter(tree)

    def get_tree_no_parents_no_content(self, empty_tree, converter=None):
        """Make a tree with no parents and no contents from empty_tree.

        :param empty_tree: A working tree with no content and no parents to
            modify.
        """
        if empty_tree.supports_setting_file_ids():
            empty_tree.set_root_id(b'empty-root-id')
        return self._convert_tree(empty_tree, converter)

    def _make_abc_tree(self, tree):
        """setup an abc content tree."""
        files = ['a', 'b/', 'b/c']
        self.build_tree(files, line_endings='binary', transport=tree.controldir.root_transport)
        tree.add(files)

    def get_tree_no_parents_abc_content(self, tree, converter=None):
        """return a test tree with a, b/, b/c contents."""
        self._make_abc_tree(tree)
        return self._convert_tree(tree, converter)

    def get_tree_no_parents_abc_content_2(self, tree, converter=None):
        """return a test tree with a, b/, b/c contents.

        This variation changes the content of 'a' to foobar
.
        """
        self._make_abc_tree(tree)
        with open(tree.basedir + '/a', 'wb') as f:
            f.write(b'foobar\n')
        return self._convert_tree(tree, converter)

    def get_tree_no_parents_abc_content_3(self, tree, converter=None):
        """return a test tree with a, b/, b/c contents.

        This variation changes the executable flag of b/c to True.
        """
        self._make_abc_tree(tree)
        tt = tree.transform()
        trans_id = tt.trans_id_tree_path('b/c')
        tt.set_executability(True, trans_id)
        tt.apply()
        return self._convert_tree(tree, converter)

    def get_tree_no_parents_abc_content_4(self, tree, converter=None):
        """return a test tree with d, b/, b/c contents.

        This variation renames a to d.
        """
        self._make_abc_tree(tree)
        tree.rename_one('a', 'd')
        return self._convert_tree(tree, converter)

    def get_tree_no_parents_abc_content_5(self, tree, converter=None):
        """return a test tree with d, b/, b/c contents.

        This variation renames a to d and alters its content to 'bar
'.
        """
        self._make_abc_tree(tree)
        tree.rename_one('a', 'd')
        with open(tree.basedir + '/d', 'wb') as f:
            f.write(b'bar\n')
        return self._convert_tree(tree, converter)

    def get_tree_no_parents_abc_content_6(self, tree, converter=None):
        """return a test tree with a, b/, e contents.

        This variation renames b/c to e, and makes it executable.
        """
        self._make_abc_tree(tree)
        tt = tree.transform()
        trans_id = tt.trans_id_tree_path('b/c')
        parent_trans_id = tt.trans_id_tree_path('')
        tt.adjust_path('e', parent_trans_id, trans_id)
        tt.set_executability(True, trans_id)
        tt.apply()
        return self._convert_tree(tree, converter)

    def get_tree_no_parents_abc_content_7(self, tree, converter=None):
        """return a test tree with a, b/, d/e contents.

        This variation adds a dir 'd' (b'd-id'), renames b to d/e.
        """
        self._make_abc_tree(tree)
        self.build_tree(['d/'], transport=tree.controldir.root_transport)
        tree.add(['d'])
        tt = tree.transform()
        trans_id = tt.trans_id_tree_path('b')
        parent_trans_id = tt.trans_id_tree_path('d')
        tt.adjust_path('e', parent_trans_id, trans_id)
        tt.apply()
        return self._convert_tree(tree, converter)

    def get_tree_with_subdirs_and_all_content_types(self):
        """Return a test tree with subdirs and all content types.
        See get_tree_with_subdirs_and_all_supported_content_types for details.
        """
        return self.get_tree_with_subdirs_and_all_supported_content_types(True)

    def get_tree_with_subdirs_and_all_supported_content_types(self, symlinks):
        """Return a test tree with subdirs and all supported content types.
        Some content types may not be created on some platforms
        (like symlinks on native win32)

        :param  symlinks:   control is symlink should be created in the tree.
                            Note: if you wish to automatically set this
                            parameters depending on underlying system,
                            please use value returned
                            by breezy.osutils.supports_symlinks() function.

        The returned tree has the following inventory:
            ['',
             '0file',
             '1top-dir',
             u'2utfሴfile',
             'symlink',            # only if symlinks arg is True
             '1top-dir/0file-in-1topdir',
             '1top-dir/1dir-in-1topdir']
        where each component has the type of its name -
        i.e. '1file..' is afile.

        note that the order of the paths and fileids is deliberately
        mismatched to ensure that the result order is path based.
        """
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('.')
        paths = ['0file', '1top-dir/', '2utfሴfile', '1top-dir/0file-in-1topdir', '1top-dir/1dir-in-1topdir/']
        self.build_tree(paths)
        tree.add(paths)
        tt = tree.transform()
        if symlinks:
            root_transaction_id = tt.trans_id_tree_path('')
            tt.new_symlink('symlink', root_transaction_id, 'link-target', b'symlink')
        tt.apply()
        return self.workingtree_to_test_tree(tree)