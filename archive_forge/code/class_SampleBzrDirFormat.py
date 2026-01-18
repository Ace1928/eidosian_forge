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
class SampleBzrDirFormat(bzrdir.BzrDirFormat):
    """A sample format

    this format is initializable, unsupported to aid in testing the
    open and open_downlevel routines.
    """

    def get_format_string(self):
        """See BzrDirFormat.get_format_string()."""
        return b'Sample .bzr dir format.'

    def initialize_on_transport(self, t):
        """Create a bzr dir."""
        t.mkdir('.bzr')
        t.put_bytes('.bzr/branch-format', self.get_format_string())
        return SampleBzrDir(t, self)

    def is_supported(self):
        return False

    def open(self, transport, _found=None):
        return 'opened branch.'

    @classmethod
    def from_string(cls, format_string):
        return cls()