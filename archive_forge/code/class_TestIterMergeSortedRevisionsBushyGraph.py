from breezy import errors, revision, tests
from breezy.tests import per_branch
class TestIterMergeSortedRevisionsBushyGraph(per_branch.TestCaseWithBranch):

    def setUp(self):
        super().setUp()
        self.revids = {}

    def make_branch_builder(self, relpath):
        try:
            builder = super().make_branch_builder(relpath)
        except (errors.TransportNotPossible, errors.UninitializableFormat):
            raise tests.TestNotApplicable('format not directly constructable')
        return builder

    def make_snapshot(self, builder, parents, revid_name):
        self.assertNotIn(revid_name, self.revids)
        if parents is None:
            files = [('add', ('', None, 'directory', ''))]
        else:
            parents = [self.revids[name] for name in parents]
            files = []
        self.revids[revid_name] = builder.build_snapshot(parents, files, message='Revision %s' % revid_name)

    def make_branch_with_embedded_merges(self, relpath='.'):
        builder = self.make_branch_builder(relpath)
        builder.start_series()
        self.make_snapshot(builder, None, '1')
        self.make_snapshot(builder, ['1'], '1.1.1')
        self.make_snapshot(builder, ['1', '1.1.1'], '2')
        self.make_snapshot(builder, ['2'], '2.1.1')
        self.make_snapshot(builder, ['2.1.1'], '2.1.2')
        self.make_snapshot(builder, ['2.1.1'], '2.2.1')
        self.make_snapshot(builder, ['2.1.2', '2.2.1'], '2.1.3')
        self.make_snapshot(builder, ['2'], '3')
        self.make_snapshot(builder, ['3', '2.1.3'], '4')
        builder.finish_series()
        br = builder.get_branch()
        br.lock_read()
        self.addCleanup(br.unlock)
        return br

    def make_branch_with_different_depths_merges(self, relpath='.'):
        builder = self.make_branch_builder(relpath)
        builder.start_series()
        self.make_snapshot(builder, None, '1')
        self.make_snapshot(builder, ['1'], '2')
        self.make_snapshot(builder, ['1'], '1.1.1')
        self.make_snapshot(builder, ['1.1.1'], '1.1.2')
        self.make_snapshot(builder, ['1.1.1'], '1.2.1')
        self.make_snapshot(builder, ['1.2.1'], '1.2.2')
        self.make_snapshot(builder, ['1.2.1'], '1.3.1')
        self.make_snapshot(builder, ['1.3.1'], '1.3.2')
        self.make_snapshot(builder, ['1.3.1'], '1.4.1')
        self.make_snapshot(builder, ['1.3.2'], '1.3.3')
        self.make_snapshot(builder, ['1.2.2', '1.3.3'], '1.2.3')
        self.make_snapshot(builder, ['2'], '2.1.1')
        self.make_snapshot(builder, ['2.1.1'], '2.1.2')
        self.make_snapshot(builder, ['2.1.1'], '2.2.1')
        self.make_snapshot(builder, ['2.1.2', '2.2.1'], '2.1.3')
        self.make_snapshot(builder, ['2', '1.2.3'], '3')
        self.make_snapshot(builder, ['3', '2.1.3'], '4')
        builder.finish_series()
        br = builder.get_branch()
        br.lock_read()
        self.addCleanup(br.unlock)
        return br

    def make_branch_with_alternate_ancestries(self, relpath='.'):
        builder = self.make_branch_builder(relpath)
        builder.start_series()
        self.make_snapshot(builder, None, '1')
        self.make_snapshot(builder, ['1'], '1.1.1')
        self.make_snapshot(builder, ['1', '1.1.1'], '2')
        self.make_snapshot(builder, ['1.1.1'], '1.2.1')
        self.make_snapshot(builder, ['1.1.1', '1.2.1'], '1.1.2')
        self.make_snapshot(builder, ['2', '1.1.2'], '3')
        builder.finish_series()
        br = builder.get_branch()
        br.lock_read()
        self.addCleanup(br.unlock)
        return br

    def assertIterRevids(self, expected, branch, *args, **kwargs):
        if kwargs.get('stop_revision_id') is not None:
            kwargs['stop_revision_id'] = self.revids[kwargs['stop_revision_id']]
        if kwargs.get('start_revision_id') is not None:
            kwargs['start_revision_id'] = self.revids[kwargs['start_revision_id']]
        revids = [revid for revid, depth, revno, eom in branch.iter_merge_sorted_revisions(*args, **kwargs)]
        self.assertEqual([self.revids[short] for short in expected], revids)

    def test_merge_sorted_starting_at_embedded_merge(self):
        branch = self.make_branch_with_embedded_merges()
        self.assertIterRevids(['4', '2.1.3', '2.2.1', '2.1.2', '2.1.1', '3', '2', '1.1.1', '1'], branch)
        self.assertIterRevids(['2.2.1', '2.1.1', '2', '1.1.1', '1'], branch, start_revision_id='2.2.1', stop_rule='with-merges')

    def test_merge_sorted_with_different_depths_merge(self):
        branch = self.make_branch_with_different_depths_merges()
        self.assertIterRevids(['4', '2.1.3', '2.2.1', '2.1.2', '2.1.1', '3', '1.2.3', '1.3.3', '1.3.2', '1.3.1', '1.2.2', '1.2.1', '1.1.1', '2', '1'], branch)
        self.assertIterRevids(['2.2.1', '2.1.1', '2', '1'], branch, start_revision_id='2.2.1', stop_rule='with-merges')

    def test_merge_sorted_exclude_ancestry(self):
        branch = self.make_branch_with_alternate_ancestries()
        self.assertIterRevids(['3', '1.1.2', '1.2.1', '2', '1.1.1', '1'], branch)
        self.assertIterRevids(['1.1.2', '1.2.1'], branch, stop_rule='with-merges-without-common-ancestry', start_revision_id='1.1.2', stop_revision_id='1.1.1')