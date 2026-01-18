import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def _test_change_file_into_dir(self, file_name):
    self.make_branch_and_working_tree()
    self.add_file(file_name, b'foo')
    self.do_full_upload()
    self.transform_file_into_dir(file_name)
    fpath = '{}/{}'.format(file_name, 'file')
    self.add_file(fpath, b'bar')
    self.assertUpFileEqual(b'foo', file_name)
    self.do_upload()
    self.assertUpFileEqual(b'bar', fpath)