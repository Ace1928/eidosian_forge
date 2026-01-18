import gzip
import os
import time
from io import BytesIO
from dulwich import porcelain
from dulwich.errors import HangupException
from dulwich.repo import Repo as GitRepo
from ...branch import Branch
from ...controldir import BranchReferenceLoop, ControlDir
from ...errors import (ConnectionReset, DivergedBranches, NoSuchTag,
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import ExecutableFeature
from ...urlutils import join as urljoin
from ..mapping import default_mapping
from ..remote import (GitRemoteRevisionTree, GitSmartRemoteNotSupported,
from ..tree import MissingNestedTree
class ParseHangupTests(TestCase):

    def setUp(self):
        super().setUp()
        try:
            HangupException([b'foo'])
        except TypeError:
            self.skipTest('dulwich version too old')

    def test_not_set(self):
        self.assertIsInstance(parse_git_hangup('http://', HangupException()), ConnectionReset)

    def test_single_line(self):
        self.assertEqual(RemoteGitError('foo bar'), parse_git_hangup('http://', HangupException([b'foo bar'])))

    def test_multi_lines(self):
        self.assertEqual(RemoteGitError('foo bar\nbla bla'), parse_git_hangup('http://', HangupException([b'foo bar', b'bla bla'])))

    def test_filter_boring(self):
        self.assertEqual(RemoteGitError('foo bar'), parse_git_hangup('http://', HangupException([b'=======', b'foo bar', b'======'])))
        self.assertEqual(RemoteGitError('foo bar'), parse_git_hangup('http://', HangupException([b'remote: =======', b'remote: foo bar', b'remote: ======'])))

    def test_permission_denied(self):
        self.assertEqual(PermissionDenied('http://', 'You are not allowed to push code to this project.'), parse_git_hangup('http://', HangupException([b'=======', b'You are not allowed to push code to this project.', b'', b'======'])))

    def test_notbrancherror_yet(self):
        self.assertEqual(NotBranchError('http://', 'A repository for this project does not exist yet.'), parse_git_hangup('http://', HangupException([b'=======', b'', b'A repository for this project does not exist yet.', b'', b'======'])))