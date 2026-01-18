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
class TestBzrDirHooks(TestCaseWithMemoryTransport):

    def test_pre_open_called(self):
        calls = []
        bzrdir.BzrDir.hooks.install_named_hook('pre_open', calls.append, None)
        transport = self.get_transport('foo')
        url = transport.base
        self.assertRaises(errors.NotBranchError, bzrdir.BzrDir.open, url)
        self.assertEqual([transport.base], [t.base for t in calls])

    def test_pre_open_actual_exceptions_raised(self):
        count = [0]

        def fail_once(transport):
            count[0] += 1
            if count[0] == 1:
                raise errors.BzrError('fail')
        bzrdir.BzrDir.hooks.install_named_hook('pre_open', fail_once, None)
        transport = self.get_transport('foo')
        url = transport.base
        err = self.assertRaises(errors.BzrError, bzrdir.BzrDir.open, url)
        self.assertEqual('fail', err._preformatted_string)

    def test_post_repo_init(self):
        from ...controldir import RepoInitHookParams
        calls = []
        bzrdir.BzrDir.hooks.install_named_hook('post_repo_init', calls.append, None)
        self.make_repository('foo')
        self.assertLength(1, calls)
        params = calls[0]
        self.assertIsInstance(params, RepoInitHookParams)
        self.assertTrue(hasattr(params, 'controldir'))
        self.assertTrue(hasattr(params, 'repository'))

    def test_post_repo_init_hook_repr(self):
        param_reprs = []
        bzrdir.BzrDir.hooks.install_named_hook('post_repo_init', lambda params: param_reprs.append(repr(params)), None)
        self.make_repository('foo')
        self.assertLength(1, param_reprs)
        param_repr = param_reprs[0]
        self.assertStartsWith(param_repr, '<RepoInitHookParams for ')