from breezy import branch, delta, errors, revision, transport
from breezy.tests import per_branch
class TestCommitHook(per_branch.TestCaseWithBranch):

    def setUp(self):
        self.hook_calls = []
        super().setUp()

    def capture_post_commit_hook(self, local, master, old_revno, old_revid, new_revno, new_revid):
        """Capture post commit hook calls to self.hook_calls.

        The call is logged, as is some state of the two branches.
        """
        if local:
            local_locked = local.is_locked()
            local_base = local.base
        else:
            local_locked = None
            local_base = None
        self.hook_calls.append(('post_commit', local_base, master.base, old_revno, old_revid, new_revno, new_revid, local_locked, master.is_locked()))

    def capture_pre_commit_hook(self, local, master, old_revno, old_revid, new_revno, new_revid, tree_delta, future_tree):
        self.hook_calls.append(('pre_commit', old_revno, old_revid, new_revno, new_revid, tree_delta))

    def test_post_commit_to_origin(self):
        tree = self.make_branch_and_memory_tree('branch')
        branch.Branch.hooks.install_named_hook('post_commit', self.capture_post_commit_hook, None)
        tree.lock_write()
        tree.add('')
        revid = tree.commit('a revision')
        self.assertEqual([('post_commit', None, tree.branch.base, 0, revision.NULL_REVISION, 1, revid, None, True)], self.hook_calls)
        tree.unlock()

    def test_post_commit_bound(self):
        master = self.make_branch('master')
        tree = self.make_branch_and_memory_tree('local')
        try:
            tree.branch.bind(master)
        except branch.BindingUnsupported:
            return
        branch.Branch.hooks.install_named_hook('post_commit', self.capture_post_commit_hook, None)
        tree.lock_write()
        tree.add('')
        revid = tree.commit('a revision')
        self.assertEqual([('post_commit', tree.branch.base, master.base, 0, revision.NULL_REVISION, 1, revid, True, True)], self.hook_calls)
        tree.unlock()

    def test_post_commit_not_to_origin(self):
        tree = self.make_branch_and_memory_tree('branch')
        with tree.lock_write():
            tree.add('')
            revid = tree.commit('first revision')
            branch.Branch.hooks.install_named_hook('post_commit', self.capture_post_commit_hook, None)
            revid2 = tree.commit('second revision')
            self.assertEqual([('post_commit', None, tree.branch.base, 1, revid, 2, revid2, None, True)], self.hook_calls)

    def get_rootfull_delta(self, repository, revid):
        tree = repository.revision_tree(revid)
        with repository.lock_read():
            parent_revid = repository.get_parent_map([revid])[revid][0]
            basis_tree = repository.revision_tree(parent_revid)
            tree = repository.revision_tree(revid)
            return tree.changes_from(basis_tree, include_root=True)

    def test_pre_commit_passes(self):
        tree = self.make_branch_and_memory_tree('branch')
        with tree.lock_write():
            tree.add('')
            branch.Branch.hooks.install_named_hook('pre_commit', self.capture_pre_commit_hook, None)
            revid1 = tree.commit('first revision')
            revid2 = tree.commit('second revision')
            root_delta = self.get_rootfull_delta(tree.branch.repository, revid1)
            empty_delta = tree.branch.repository.get_revision_delta(revid2)
            self.assertEqual([('pre_commit', 0, revision.NULL_REVISION, 1, revid1, root_delta), ('pre_commit', 1, revid1, 2, revid2, empty_delta)], self.hook_calls)

    def test_pre_commit_fails(self):
        tree = self.make_branch_and_memory_tree('branch')
        with tree.lock_write():
            tree.add('')

            class PreCommitException(Exception):

                def __init__(self, revid):
                    self.revid = revid

            def hook_func(local, master, old_revno, old_revid, new_revno, new_revid, tree_delta, future_tree):
                raise PreCommitException(new_revid)
            branch.Branch.hooks.install_named_hook('pre_commit', self.capture_pre_commit_hook, None)
            branch.Branch.hooks.install_named_hook('pre_commit', hook_func, None)
            revids = [None, None, None]
            err = self.assertRaises(PreCommitException, tree.commit, 'message')
            revids[0] = err.revid
            branch.Branch.hooks['pre_commit'] = []
            branch.Branch.hooks.install_named_hook('pre_commit', self.capture_pre_commit_hook, None)
            for i in range(1, 3):
                revids[i] = tree.commit('message')
            self.assertEqual([('pre_commit', 0, revision.NULL_REVISION, 1, revids[0], self.get_rootfull_delta(tree.branch.repository, revids[0])), ('pre_commit', 0, revision.NULL_REVISION, 1, revids[1], self.get_rootfull_delta(tree.branch.repository, revids[1])), ('pre_commit', 1, revids[1], 2, revids[2], self.get_rootfull_delta(tree.branch.repository, revids[2]))], self.hook_calls)

    def test_pre_commit_delta(self):
        self.build_tree(['rootfile', 'dir/', 'dir/subfile'])
        tree = self.make_branch_and_tree('.')
        with tree.lock_write():
            tree.add('rootfile')
            tree.put_file_bytes_non_atomic('rootfile', b'abc')
            tree.add('dir')
            tree.add('dir/subfile')
            tree.put_file_bytes_non_atomic('to_be_unversioned', b'blah')
            tree.add(['to_be_unversioned'])
            tree.put_file_bytes_non_atomic('dir/subfile', b'def')
            revid1 = tree.commit('first revision')
        with tree.lock_write():
            tree.put_file_bytes_non_atomic('rootfile', b'jkl')
            tree.rename_one('dir/subfile', 'dir/subfile_renamed')
            tree.unversion(['to_be_unversioned'])
            tree.mkdir('added_dir')
            branch.Branch.hooks.install_named_hook('pre_commit', self.capture_pre_commit_hook, None)
            revid2 = tree.commit('second revision')
        self.assertEqual([('pre_commit', 1, revid1, 2, revid2, self.get_rootfull_delta(tree.branch.repository, revid2))], self.hook_calls)