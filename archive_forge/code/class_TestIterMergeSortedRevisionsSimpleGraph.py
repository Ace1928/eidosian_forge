from breezy import errors, revision, tests
from breezy.tests import per_branch
class TestIterMergeSortedRevisionsSimpleGraph(per_branch.TestCaseWithBranch):

    def setUp(self):
        super().setUp()
        self.revids = {}
        builder = self.make_builder_with_merges('.')
        self.branch = builder.get_branch()
        self.branch.lock_read()
        self.addCleanup(self.branch.unlock)

    def make_snapshot(self, builder, parents, revid_name):
        self.assertNotIn(revid_name, self.revids)
        if parents is None:
            files = [('add', ('', None, 'directory', ''))]
        else:
            parents = [self.revids[name] for name in parents]
            files = []
        self.revids[revid_name] = builder.build_snapshot(parents, files, message='Revision %s' % revid_name)

    def make_builder_with_merges(self, relpath):
        try:
            builder = self.make_branch_builder(relpath)
        except (errors.TransportNotPossible, errors.UninitializableFormat):
            raise tests.TestNotApplicable('format not directly constructable')
        builder.start_series()
        self.make_snapshot(builder, None, '1')
        self.make_snapshot(builder, ['1'], '1.1.1')
        self.make_snapshot(builder, ['1'], '2')
        self.make_snapshot(builder, ['2', '1.1.1'], '3')
        builder.finish_series()
        return builder

    def assertIterRevids(self, expected, *args, **kwargs):
        if kwargs.get('stop_revision_id') is not None:
            kwargs['stop_revision_id'] = self.revids[kwargs['stop_revision_id']]
        if kwargs.get('start_revision_id') is not None:
            kwargs['start_revision_id'] = self.revids[kwargs['start_revision_id']]
        revids = [revid for revid, depth, revno, eom in self.branch.iter_merge_sorted_revisions(*args, **kwargs)]
        self.assertEqual([self.revids[short] for short in expected], revids)

    def test_merge_sorted(self):
        self.assertIterRevids(['3', '1.1.1', '2', '1'])

    def test_merge_sorted_range(self):
        self.assertIterRevids(['1.1.1'], start_revision_id='1.1.1', stop_revision_id='1')

    def test_merge_sorted_range_start_only(self):
        self.assertIterRevids(['1.1.1', '1'], start_revision_id='1.1.1')

    def test_merge_sorted_range_stop_exclude(self):
        self.assertIterRevids(['3', '1.1.1', '2'], stop_revision_id='1')

    def test_merge_sorted_range_stop_include(self):
        self.assertIterRevids(['3', '1.1.1', '2'], stop_revision_id='2', stop_rule='include')

    def test_merge_sorted_range_stop_with_merges(self):
        self.assertIterRevids(['3', '1.1.1'], stop_revision_id='3', stop_rule='with-merges')

    def test_merge_sorted_range_stop_with_merges_can_show_non_parents(self):
        self.assertIterRevids(['3', '1.1.1', '2'], stop_revision_id='2', stop_rule='with-merges')

    def test_merge_sorted_range_stop_with_merges_ignore_non_parents(self):
        self.assertIterRevids(['3', '1.1.1'], stop_revision_id='1.1.1', stop_rule='with-merges')

    def test_merge_sorted_single_stop_exclude(self):
        self.assertIterRevids([], start_revision_id='3', stop_revision_id='3')

    def test_merge_sorted_single_stop_include(self):
        self.assertIterRevids(['3'], start_revision_id='3', stop_revision_id='3', stop_rule='include')

    def test_merge_sorted_single_stop_with_merges(self):
        self.assertIterRevids(['3', '1.1.1'], start_revision_id='3', stop_revision_id='3', stop_rule='with-merges')

    def test_merge_sorted_forward(self):
        self.assertIterRevids(['1', '2', '1.1.1', '3'], direction='forward')

    def test_merge_sorted_range_forward(self):
        self.assertIterRevids(['1.1.1'], start_revision_id='1.1.1', stop_revision_id='1', direction='forward')

    def test_merge_sorted_range_start_only_forward(self):
        self.assertIterRevids(['1', '1.1.1'], start_revision_id='1.1.1', direction='forward')

    def test_merge_sorted_range_stop_exclude_forward(self):
        self.assertIterRevids(['2', '1.1.1', '3'], stop_revision_id='1', direction='forward')

    def test_merge_sorted_range_stop_include_forward(self):
        self.assertIterRevids(['2', '1.1.1', '3'], stop_revision_id='2', stop_rule='include', direction='forward')

    def test_merge_sorted_range_stop_with_merges_forward(self):
        self.assertIterRevids(['1.1.1', '3'], stop_revision_id='3', stop_rule='with-merges', direction='forward')