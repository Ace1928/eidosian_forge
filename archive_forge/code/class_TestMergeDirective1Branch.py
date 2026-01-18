import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
class TestMergeDirective1Branch(tests.TestCaseWithTransport, TestMergeDirectiveBranch):
    """Test merge directive format 1 with a branch"""
    EMAIL1 = EMAIL1
    EMAIL2 = EMAIL2

    def from_objects(self, repository, revision_id, time, timezone, target_branch, patch_type='bundle', local_target_branch=None, public_branch=None, message=None, base_revision_id=None):
        if base_revision_id is not None:
            raise tests.TestNotApplicable('This format does not support explicit bases.')
        with repository.lock_write():
            return merge_directive.MergeDirective.from_objects(repository, revision_id, time, timezone, target_branch, patch_type, local_target_branch, public_branch, message)

    def make_merge_directive(self, revision_id, testament_sha1, time, timezone, target_branch, patch=None, patch_type=None, source_branch=None, message=None):
        return merge_directive.MergeDirective(revision_id, testament_sha1, time, timezone, target_branch, patch, patch_type, source_branch, message)