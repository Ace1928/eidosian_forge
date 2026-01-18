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
class GitUrlAndPathFromTransportTests(TestCase):

    def test_file(self):
        split_url = _git_url_and_path_from_transport('file:///home/blah')
        self.assertEqual(split_url.scheme, 'file')
        self.assertEqual(split_url.path, '/home/blah')

    def test_file_segment_params(self):
        split_url = _git_url_and_path_from_transport('file:///home/blah,branch=master')
        self.assertEqual(split_url.scheme, 'file')
        self.assertEqual(split_url.path, '/home/blah')

    def test_git_smart(self):
        split_url = _git_url_and_path_from_transport('git://github.com/dulwich/dulwich,branch=master')
        self.assertEqual(split_url.scheme, 'git')
        self.assertEqual(split_url.path, '/dulwich/dulwich')

    def test_https(self):
        split_url = _git_url_and_path_from_transport('https://github.com/dulwich/dulwich')
        self.assertEqual(split_url.scheme, 'https')
        self.assertEqual(split_url.path, '/dulwich/dulwich')

    def test_https_segment_params(self):
        split_url = _git_url_and_path_from_transport('https://github.com/dulwich/dulwich,branch=master')
        self.assertEqual(split_url.scheme, 'https')
        self.assertEqual(split_url.path, '/dulwich/dulwich')