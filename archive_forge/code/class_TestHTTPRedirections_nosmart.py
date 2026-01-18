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
class TestHTTPRedirections_nosmart(TestHTTPRedirectionsBase, http_utils.TestCaseWithTwoWebservers):
    """Tests redirections for the nosmart decorator"""
    _transport = NoSmartTransportDecorator

    def _qualified_url(self, host, port):
        result = 'nosmart+http://{}:{}'.format(host, port)
        self.permit_url(result)
        return result