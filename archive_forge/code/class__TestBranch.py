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
class _TestBranch(breezy.branch.Branch):
    """Test Branch implementation for TestBzrDirSprout."""

    @property
    def control_transport(self):
        return self._transport

    def __init__(self, transport, *args, **kwargs):
        self._format = _TestBranchFormat()
        self._transport = transport
        self.base = transport.base
        super().__init__(*args, **kwargs)
        self.calls = []
        self._parent = None

    def sprout(self, *args, **kwargs):
        self.calls.append('sprout')
        return _TestBranch(self._transport)

    def copy_content_into(self, destination, revision_id=None):
        self.calls.append('copy_content_into')

    def last_revision(self):
        return _mod_revision.NULL_REVISION

    def get_parent(self):
        return self._parent

    def _get_config(self):
        return config.TransportConfig(self._transport, 'branch.conf')

    def _get_config_store(self):
        return config.BranchStore(self)

    def set_parent(self, parent):
        self._parent = parent

    def lock_read(self):
        return lock.LogicalLockResult(self.unlock)

    def unlock(self):
        return