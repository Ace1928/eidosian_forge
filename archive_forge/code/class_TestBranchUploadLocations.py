import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
class TestBranchUploadLocations(per_branch.TestCaseWithBranch):

    def test_get_upload_location_unset(self):
        conf = self.get_branch().get_config_stack()
        self.assertEqual(None, conf.get('upload_location'))

    def test_get_push_location_exact(self):
        bedding.ensure_config_dir_exists()
        fn = bedding.locations_config_path()
        b = self.get_branch()
        with open(fn, 'w') as f:
            f.write('[%s]\nupload_location=foo\n' % b.base.rstrip('/'))
        self.assertEqual('foo', b.get_config_stack().get('upload_location'))

    def test_set_push_location(self):
        conf = self.get_branch().get_config_stack()
        conf.set('upload_location', 'foo')
        self.assertEqual('foo', conf.get('upload_location'))