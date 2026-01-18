import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
class TestFullUpload(tests.TestCaseWithTransport, TestUploadMixin):
    do_upload = TestUploadMixin.do_full_upload

    def test_full_upload_empty_tree(self):
        self.make_branch_and_working_tree()
        self.do_full_upload()
        revid_path = self.tree.branch.get_config_stack().get('upload_revid_location')
        self.assertUpPathExists(revid_path)

    def test_invalid_revspec(self):
        self.make_branch_and_working_tree()
        rev1 = revisionspec.RevisionSpec.from_string('1')
        rev2 = revisionspec.RevisionSpec.from_string('2')
        self.assertRaises(errors.CommandError, self.do_incremental_upload, revision=[rev1, rev2])

    def test_create_remote_dir_twice(self):
        self.make_branch_and_working_tree()
        self.add_dir('dir')
        self.do_full_upload()
        self.add_file('dir/goodbye', b'baz')
        self.assertUpPathDoesNotExist('dir/goodbye')
        self.do_full_upload()
        self.assertUpFileEqual(b'baz', 'dir/goodbye')
        self.assertUpPathModeEqual('dir', 509)