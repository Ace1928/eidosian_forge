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
class SampleBzrDir(bzrdir.BzrDir):
    """A sample BzrDir implementation to allow testing static methods."""

    def create_repository(self, shared=False):
        """See ControlDir.create_repository."""
        return 'A repository'

    def open_repository(self):
        """See ControlDir.open_repository."""
        return SampleRepository(self)

    def create_branch(self, name=None):
        """See ControlDir.create_branch."""
        if name is not None:
            raise controldir.NoColocatedBranchSupport(self)
        return SampleBranch(self)

    def create_workingtree(self):
        """See ControlDir.create_workingtree."""
        return 'A tree'