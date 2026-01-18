from breezy.tests.per_branch import TestCaseWithBranch
class TestCaching(TestCaseWithBranch):
    """Tests for the caching of branches' dotted revno generation.

    When locked, branches should avoid regenerating revision_id=>dotted revno
    mapping.

    When not locked, obviously the revision_id => dotted revno will need to be
    regenerated or reread each time.

    We test if revision_history is using the cache by instrumenting the branch's
    _gen_revno_map method, which is called by get_revision_id_to_revno_map.
    """

    def get_instrumented_branch(self):
        """Get a branch and monkey patch it to log calls to _gen_revno_map.

        :returns: a tuple of (the branch, list that calls will be logged to)
        """
        tree, revmap = self.create_tree_with_merge()
        calls = []
        real_func = tree.branch._gen_revno_map

        def wrapper():
            calls.append('_gen_revno_map')
            return real_func()
        tree.branch._gen_revno_map = wrapper
        return (tree.branch, revmap, calls)

    def test_unlocked(self):
        """Repeated calls will call _gen_revno_map each time."""
        branch, revmap, calls = self.get_instrumented_branch()
        branch.get_revision_id_to_revno_map()
        branch.get_revision_id_to_revno_map()
        branch.get_revision_id_to_revno_map()
        self.assertEqual(['_gen_revno_map'] * 3, calls)

    def test_locked(self):
        """Repeated calls will only call _gen_revno_map once.
        """
        branch, revmap, calls = self.get_instrumented_branch()
        with branch.lock_read():
            branch.get_revision_id_to_revno_map()
            self.assertEqual(['_gen_revno_map'], calls)

    def test_set_last_revision_info_when_locked(self):
        """Calling set_last_revision_info should reset the cache."""
        branch, revmap, calls = self.get_instrumented_branch()
        with branch.lock_write():
            self.assertEqual({revmap['1']: (1,), revmap['2']: (2,), revmap['3']: (3,), revmap['1.1.1']: (1, 1, 1)}, branch.get_revision_id_to_revno_map())
            branch.set_last_revision_info(2, revmap['2'])
            self.assertEqual({revmap['1']: (1,), revmap['2']: (2,)}, branch.get_revision_id_to_revno_map())
            self.assertEqual({revmap['1']: (1,), revmap['2']: (2,)}, branch.get_revision_id_to_revno_map())
            self.assertEqual(['_gen_revno_map'] * 2, calls)