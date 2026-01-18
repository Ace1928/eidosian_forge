import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
class TestMergeDirective2Branch(tests.TestCaseWithTransport, TestMergeDirectiveBranch):
    """Test merge directive format 2 with a branch"""
    EMAIL1 = EMAIL1_2
    EMAIL2 = EMAIL2_2

    def from_objects(self, repository, revision_id, time, timezone, target_branch, patch_type='bundle', local_target_branch=None, public_branch=None, message=None, base_revision_id=None):
        include_patch = patch_type in ('bundle', 'diff')
        include_bundle = patch_type == 'bundle'
        self.assertTrue(patch_type in ('bundle', 'diff', None))
        return merge_directive.MergeDirective2.from_objects(repository, revision_id, time, timezone, target_branch, include_patch, include_bundle, local_target_branch, public_branch, message, base_revision_id)

    def make_merge_directive(self, revision_id, testament_sha1, time, timezone, target_branch, patch=None, patch_type=None, source_branch=None, message=None, base_revision_id=b'null:'):
        if patch_type == 'bundle':
            bundle = patch
            patch = None
        else:
            bundle = None
        return merge_directive.MergeDirective2(revision_id, testament_sha1, time, timezone, target_branch, patch, source_branch, message, bundle, base_revision_id)

    def test_base_revision(self):
        tree_a, tree_b, branch_c = self.make_trees()
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 60, tree_b.branch.base, patch_type='bundle', public_branch=tree_a.branch.base, base_revision_id=None)
        self.assertEqual(b'rev1', md.base_revision_id)
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 60, tree_b.branch.base, patch_type='bundle', public_branch=tree_a.branch.base, base_revision_id=b'null:')
        self.assertEqual(b'null:', md.base_revision_id)
        lines = md.to_lines()
        md2 = merge_directive.MergeDirective.from_lines(lines)
        self.assertEqual(md2.base_revision_id, md.base_revision_id)

    def test_patch_verification(self):
        tree_a, tree_b, branch_c = self.make_trees()
        md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 60, tree_b.branch.base, patch_type='bundle', public_branch=tree_a.branch.base)
        lines = md.to_lines()
        md2 = merge_directive.MergeDirective.from_lines(lines)
        md2._verify_patch(tree_a.branch.repository)
        md2.patch = md2.patch.replace(b' \n', b'\n')
        md2._verify_patch(tree_a.branch.repository)
        md2.patch = re.sub(b'(\r\n|\r|\n)', b'\r', md2.patch)
        self.assertTrue(md2._verify_patch(tree_a.branch.repository))
        md2.patch = re.sub(b'(\r\n|\r|\n)', b'\r\n', md2.patch)
        self.assertTrue(md2._verify_patch(tree_a.branch.repository))
        md2.patch = md2.patch.replace(b'content_c', b'content_d')
        self.assertFalse(md2._verify_patch(tree_a.branch.repository))