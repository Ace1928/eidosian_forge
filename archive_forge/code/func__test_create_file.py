import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def _test_create_file(self, file_name):
    self.make_branch_and_working_tree()
    self.do_full_upload()
    self.add_file(file_name, b'foo')
    self.do_upload()
    self.assertUpFileEqual(b'foo', file_name)