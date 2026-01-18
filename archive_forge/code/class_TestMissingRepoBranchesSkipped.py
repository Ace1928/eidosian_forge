import os
import subprocess
import sys
import breezy.branch
import breezy.bzr.branch
from ... import (branch, bzr, config, controldir, errors, help_topics, lock,
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ... import urlutils, win32utils
from ...errors import (NotBranchError, UnknownFormatError,
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from ...transport import memory, pathfilter
from ...transport.http.urllib import HttpTransport
from ...transport.nosmart import NoSmartTransportDecorator
from ...transport.readonly import ReadonlyTransportDecorator
from .. import branch as bzrbranch
from .. import (bzrdir, knitpack_repo, knitrepo, remote, workingtree_3,
from ..fullhistory import BzrBranchFormat5
class TestMissingRepoBranchesSkipped(TestCaseWithMemoryTransport):

    def test_find_controldirs_missing_repo(self):
        t = self.get_transport()
        arepo = self.make_repository('arepo', shared=True)
        abranch_url = arepo.user_url + '/abranch'
        abranch = bzrdir.BzrDir.create(abranch_url).create_branch()
        t.delete_tree('arepo/.bzr')
        self.assertRaises(errors.NoRepositoryPresent, branch.Branch.open, abranch_url)
        self.make_branch('baz')
        for actual_bzrdir in bzrdir.BzrDir.find_branches(t):
            self.assertEndsWith(actual_bzrdir.user_url, '/baz/')