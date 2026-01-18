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
class TestHTTPRedirectionsBase:
    """Test redirection between two http servers.

    This MUST be used by daughter classes that also inherit from
    TestCaseWithTwoWebservers.

    We can't inherit directly from TestCaseWithTwoWebservers or the
    test framework will try to create an instance which cannot
    run, its implementation being incomplete.
    """

    def create_transport_readonly_server(self):
        return http_utils.HTTPServerRedirecting()

    def create_transport_secondary_server(self):
        return http_utils.HTTPServerRedirecting()

    def setUp(self):
        super().setUp()
        self.new_server = self.get_readonly_server()
        self.old_server = self.get_secondary_server()
        self.old_server.redirect_to(self.new_server.host, self.new_server.port)

    def test_loop(self):
        self.new_server.redirect_to(self.old_server.host, self.old_server.port)
        old_url = self._qualified_url(self.old_server.host, self.old_server.port)
        oldt = self._transport(old_url)
        self.assertRaises(errors.NotBranchError, bzrdir.BzrDir.open_from_transport, oldt)
        new_url = self._qualified_url(self.new_server.host, self.new_server.port)
        newt = self._transport(new_url)
        self.assertRaises(errors.NotBranchError, bzrdir.BzrDir.open_from_transport, newt)

    def test_qualifier_preserved(self):
        wt = self.make_branch_and_tree('branch')
        old_url = self._qualified_url(self.old_server.host, self.old_server.port)
        start = self._transport(old_url).clone('branch')
        bdir = bzrdir.BzrDir.open_from_transport(start)
        self.assertIsInstance(bdir.root_transport, type(start))