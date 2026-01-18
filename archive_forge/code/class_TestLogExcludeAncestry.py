import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestLogExcludeAncestry(tests.TestCaseWithTransport):

    def make_branch_with_alternate_ancestries(self, relpath='.'):
        builder = branchbuilder.BranchBuilder(self.get_transport(relpath))
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'TREE_ROOT', 'directory', ''))], revision_id=b'1')
        builder.build_snapshot([b'1'], [], revision_id=b'1.1.1')
        builder.build_snapshot([b'1'], [], revision_id=b'2')
        builder.build_snapshot([b'1.1.1'], [], revision_id=b'1.2.1')
        builder.build_snapshot([b'1.1.1', b'1.2.1'], [], revision_id=b'1.1.2')
        builder.build_snapshot([b'2', b'1.1.2'], [], revision_id=b'3')
        builder.finish_series()
        br = builder.get_branch()
        br.lock_read()
        self.addCleanup(br.unlock)
        return br

    def assertLogRevnos(self, expected_revnos, b, start, end, exclude_common_ancestry, generate_merge_revisions=True):
        iter_revs = log._calc_view_revisions(b, start, end, direction='reverse', generate_merge_revisions=generate_merge_revisions, exclude_common_ancestry=exclude_common_ancestry)
        self.assertEqual(expected_revnos, [revid for revid, revno, depth in iter_revs])

    def test_merge_sorted_exclude_ancestry(self):
        b = self.make_branch_with_alternate_ancestries()
        self.assertLogRevnos([b'3', b'1.1.2', b'1.2.1', b'1.1.1', b'2', b'1'], b, b'1', b'3', exclude_common_ancestry=False)
        self.assertLogRevnos([b'3', b'1.1.2', b'1.2.1', b'2'], b, b'1.1.1', b'3', exclude_common_ancestry=True)

    def test_merge_sorted_simple_revnos_exclude_ancestry(self):
        b = self.make_branch_with_alternate_ancestries()
        self.assertLogRevnos([b'3', b'2'], b, b'1', b'3', exclude_common_ancestry=True, generate_merge_revisions=False)
        self.assertLogRevnos([b'3', b'1.1.2', b'1.2.1', b'1.1.1', b'2'], b, b'1', b'3', exclude_common_ancestry=True, generate_merge_revisions=True)