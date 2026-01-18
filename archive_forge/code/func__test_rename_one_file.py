import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def _test_rename_one_file(self, old_name, new_name):
    self.make_branch_and_working_tree()
    self.add_file(old_name, b'foo')
    self.do_full_upload()
    self.rename_any(old_name, new_name)
    self.assertUpFileEqual(b'foo', old_name)
    self.do_upload()
    self.assertUpFileEqual(b'foo', new_name)