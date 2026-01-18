import os
import shutil
from .... import bedding, config, errors, trace
from ....merge import Merger
from ....mutabletree import MutableTree
from ....tests import TestCaseWithTransport, TestSkipped
from .. import (post_build_tree_quilt, post_merge_quilt_cleanup,
from ..merge import tree_unapply_patches
from ..quilt import QuiltPatches
from . import quilt_feature
class TestMergeHook(TestCaseWithTransport):
    _test_needs_features = [quilt_feature]

    def enable_hooks(self):
        Merger.hooks.install_named_hook('pre_merge', pre_merge_quilt, 'Debian quilt patch (un)applying and ancestry fixing')
        Merger.hooks.install_named_hook('post_merge', post_merge_quilt_cleanup, 'Cleaning up quilt temporary directories')
        MutableTree.hooks.install_named_hook('post_build_tree', post_build_tree_quilt, 'Apply quilt trees.')

    def test_diverged_patches(self):
        self.enable_hooks()
        tree_a = self.make_branch_and_tree('a')
        self.build_tree(['a/debian/', 'a/debian/patches/', 'a/debian/source/', 'a/.pc/'])
        self.build_tree_contents([('a/.pc/.quilt_patches', 'debian/patches\n'), ('a/.pc/.version', '2\n'), ('a/debian/source/format', '3.0 (quilt)'), ('a/debian/patches/series', 'patch1\n'), ('a/debian/patches/patch1', TRIVIAL_PATCH)])
        tree_a.smart_add([tree_a.basedir])
        tree_a.commit('initial')
        tree_b = tree_a.controldir.sprout('b').open_workingtree()
        self.build_tree_contents([('a/debian/patches/patch1', '\n'.join(TRIVIAL_PATCH.splitlines()[:-1] + ['+d\n']))])
        quilt_push_all(tree_a)
        tree_a.smart_add([tree_a.basedir])
        tree_a.commit('apply patches')
        self.build_tree_contents([('b/debian/patches/patch1', '\n'.join(TRIVIAL_PATCH.splitlines()[:-1] + ['+c\n']))])
        quilt_push_all(tree_b)
        tree_b.commit('apply patches')
        conflicts = tree_a.merge_from_branch(tree_b.branch)
        self.assertFileEqual('--- /dev/null\t2012-01-02 01:09:10.986490031 +0100\n+++ base/a\t2012-01-02 20:03:59.710666215 +0100\n@@ -0,0 +1 @@\n<<<<<<< TREE\n+d\n=======\n+c\n>>>>>>> MERGE-SOURCE\n', 'a/debian/patches/patch1')
        self.assertPathDoesNotExist('a/a')
        self.assertEqual(1, len(conflicts))

    def test_auto_apply_patches_after_checkout(self):
        self.enable_hooks()
        tree_a = self.make_branch_and_tree('a')
        self.build_tree(['a/debian/', 'a/debian/patches/'])
        self.build_tree_contents([('a/debian/patches/series', 'patch1\n'), ('a/debian/patches/patch1', TRIVIAL_PATCH)])
        tree_a.smart_add([tree_a.basedir])
        tree_a.commit('initial')
        bedding.ensure_config_dir_exists()
        config.GlobalStack().set('quilt.tree_policy', 'applied')
        tree_a.branch.create_checkout('b')
        self.assertFileEqual('a\n', 'b/a')

    def test_auto_apply_patches_after_update_format_1(self):
        self.enable_hooks()
        tree_a = self.make_branch_and_tree('a')
        tree_b = tree_a.branch.create_checkout('b')
        self.build_tree(['a/debian/', 'a/debian/patches/', 'a/.pc/'])
        self.build_tree_contents([('a/.pc/.quilt_patches', 'debian/patches\n'), ('a/.pc/.version', '2\n'), ('a/debian/patches/series', 'patch1\n'), ('a/debian/patches/patch1', TRIVIAL_PATCH)])
        tree_a.smart_add([tree_a.basedir])
        tree_a.commit('initial')
        self.build_tree(['b/.bzr-builddeb/', 'b/debian/', 'b/debian/source/'])
        tree_b.get_config_stack().set('quilt.tree_policy', 'applied')
        self.build_tree_contents([('b/debian/source/format', '1.0')])
        tree_b.update()
        self.assertFileEqual('a\n', 'b/a')

    def test_auto_apply_patches_after_update(self):
        self.enable_hooks()
        tree_a = self.make_branch_and_tree('a')
        tree_b = tree_a.branch.create_checkout('b')
        self.build_tree(['a/debian/', 'a/debian/patches/', 'a/debian/source/', 'a/.pc/'])
        self.build_tree_contents([('a/.pc/.quilt_patches', 'debian/patches\n'), ('a/.pc/.version', '2\n'), ('a/debian/source/format', '3.0 (quilt)'), ('a/debian/patches/series', 'patch1\n'), ('a/debian/patches/patch1', TRIVIAL_PATCH)])
        tree_a.smart_add([tree_a.basedir])
        tree_a.commit('initial')
        self.build_tree(['b/.bzr-builddeb/', 'b/debian/', 'b/debian/source/'])
        tree_b.get_config_stack().set('quilt.tree_policy', 'applied')
        self.build_tree_contents([('b/debian/source/format', '3.0 (quilt)')])
        tree_b.update()
        self.assertFileEqual('a\n', 'b/a')

    def test_auto_unapply_patches_after_update(self):
        self.enable_hooks()
        tree_a = self.make_branch_and_tree('a')
        tree_b = tree_a.branch.create_checkout('b')
        self.build_tree(['a/debian/', 'a/debian/patches/', 'a/debian/source/', 'a/.pc/'])
        self.build_tree_contents([('a/.pc/.quilt_patches', 'debian/patches\n'), ('a/.pc/.version', '2\n'), ('a/debian/source/format', '3.0 (quilt)'), ('a/debian/patches/series', 'patch1\n'), ('a/debian/patches/patch1', TRIVIAL_PATCH)])
        tree_a.smart_add([tree_a.basedir])
        tree_a.commit('initial')
        self.build_tree(['b/.bzr-builddeb/'])
        tree_b.get_config_stack().set('quilt.tree_policy', 'unapplied')
        tree_b.update()
        self.assertPathDoesNotExist('b/a')

    def test_disabled_hook(self):
        self.enable_hooks()
        tree_a = self.make_branch_and_tree('a')
        tree_a.get_config_stack().set('quilt.smart_merge', False)
        self.build_tree(['a/debian/', 'a/debian/patches/', 'a/.pc/'])
        self.build_tree_contents([('a/.pc/.quilt_patches', 'debian/patches\n'), ('a/.pc/.version', '2\n'), ('a/debian/patches/series', 'patch1\n'), ('a/debian/patches/patch1', TRIVIAL_PATCH), ('a/a', '')])
        tree_a.smart_add([tree_a.basedir])
        tree_a.commit('initial')
        tree_b = tree_a.controldir.sprout('b').open_workingtree()
        self.build_tree_contents([('a/debian/patches/patch1', '\n'.join(TRIVIAL_PATCH.splitlines()[:-1] + ['+d\n']))])
        quilt_push_all(tree_a)
        tree_a.smart_add([tree_a.basedir])
        tree_a.commit('apply patches')
        self.assertFileEqual('d\n', 'a/a')
        self.build_tree_contents([('b/debian/patches/patch1', '\n'.join(TRIVIAL_PATCH.splitlines()[:-1] + ['+c\n']))])
        quilt_push_all(tree_b)
        tree_b.commit('apply patches')
        self.assertFileEqual('c\n', 'b/a')
        conflicts = tree_a.merge_from_branch(tree_b.branch)
        self.assertFileEqual('--- /dev/null\t2012-01-02 01:09:10.986490031 +0100\n+++ base/a\t2012-01-02 20:03:59.710666215 +0100\n@@ -0,0 +1 @@\n<<<<<<< TREE\n+d\n=======\n+c\n>>>>>>> MERGE-SOURCE\n', 'a/debian/patches/patch1')
        self.assertFileEqual('<<<<<<< TREE\nd\n=======\nc\n>>>>>>> MERGE-SOURCE\n', 'a/a')
        self.assertEqual(2, len(conflicts))