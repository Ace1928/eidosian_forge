import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
class TestMergeDirective2(tests.TestCase, TestMergeDirective):
    """Test merge directive format 2"""
    INPUT1 = INPUT1_2
    OUTPUT1 = OUTPUT1_2
    OUTPUT2 = OUTPUT2_2

    def make_merge_directive(self, revision_id, testament_sha1, time, timezone, target_branch, patch=None, patch_type=None, source_branch=None, message=None, base_revision_id=b'null:'):
        if patch_type == 'bundle':
            bundle = patch
            patch = None
        else:
            bundle = None
        return merge_directive.MergeDirective2(revision_id, testament_sha1, time, timezone, target_branch, patch, source_branch, message, bundle, base_revision_id)

    @staticmethod
    def set_bundle(md, value):
        md.bundle = value