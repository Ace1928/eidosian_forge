import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def _test_make_file_executable(self, file_name):
    self.make_branch_and_working_tree()
    self.add_file(file_name, b'foo')
    self.chmod_file(file_name, 436)
    self.do_full_upload()
    self.chmod_file(file_name, 493)
    self.assertUpPathModeEqual(file_name, 436)
    self.do_upload()
    self.assertUpPathModeEqual(file_name, 509)