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
class RemoteRevisionTreeTests(TestCaseWithTransport):
    _test_needs_features = [ExecutableFeature('git')]

    def setUp(self):
        TestCaseWithTransport.setUp(self)
        self.remote_real = GitRepo.init('remote', mkdir=True)
        self.remote_url = 'git://%s/' % os.path.abspath(self.remote_real.path)
        self.permit_url(self.remote_url)
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')

    def test_open(self):
        br = Branch.open(self.remote_url)
        t = br.basis_tree()
        self.assertIsInstance(t, GitRemoteRevisionTree)
        self.assertRaises(GitSmartRemoteNotSupported, t.is_versioned, 'la')
        self.assertRaises(GitSmartRemoteNotSupported, t.has_filename, 'la')
        self.assertRaises(GitSmartRemoteNotSupported, t.get_file_text, 'la')
        self.assertRaises(GitSmartRemoteNotSupported, t.list_files, 'la')

    def test_archive(self):
        br = Branch.open(self.remote_url)
        t = br.basis_tree()
        chunks = list(t.archive('tgz', 'foo.tar.gz'))
        with gzip.GzipFile(fileobj=BytesIO(b''.join(chunks))) as g:
            self.assertEqual('', g.name)

    def test_archive_unsupported(self):
        br = Branch.open(self.remote_url)
        t = br.basis_tree()

        def raise_unsupp(*args, **kwargs):
            raise GitSmartRemoteNotSupported(raise_unsupp, None)
        self.overrideAttr(t._repository.controldir._client, 'archive', raise_unsupp)
        self.assertRaises(GitSmartRemoteNotSupported, t.archive, 'tgz', 'foo.tar.gz')