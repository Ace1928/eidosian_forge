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
class TestGenerateBackupName(TestCaseWithMemoryTransport):

    def setUp(self):
        super().setUp()
        self._transport = self.get_transport()
        bzrdir.BzrDir.create(self.get_url(), possible_transports=[self._transport])
        self._bzrdir = bzrdir.BzrDir.open_from_transport(self._transport)

    def test_new(self):
        self.assertEqual('a.~1~', self._bzrdir._available_backup_name('a'))

    def test_exiting(self):
        self._transport.put_bytes('a.~1~', b'some content')
        self.assertEqual('a.~2~', self._bzrdir._available_backup_name('a'))