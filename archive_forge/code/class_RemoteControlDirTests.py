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
class RemoteControlDirTests(TestCaseWithTransport):
    _test_needs_features = [ExecutableFeature('git')]

    def setUp(self):
        TestCaseWithTransport.setUp(self)
        self.remote_real = GitRepo.init('remote', mkdir=True)
        self.remote_url = 'git://%s/' % os.path.abspath(self.remote_real.path)
        self.permit_url(self.remote_url)

    def test_remove_branch(self):
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/heads/blah')
        remote = ControlDir.open(self.remote_url)
        remote.destroy_branch(name='blah')
        self.assertEqual(self.remote_real.get_refs(), {b'refs/heads/master': self.remote_real.head(), b'HEAD': self.remote_real.head()})

    def test_list_branches(self):
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/heads/blah')
        remote = ControlDir.open(self.remote_url)
        self.assertEqual({'master', 'blah', 'master'}, {b.name for b in remote.list_branches()})

    def test_get_branches(self):
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/heads/blah')
        remote = ControlDir.open(self.remote_url)
        self.assertEqual({'': 'master', 'blah': 'blah', 'master': 'master'}, {n: b.name for n, b in remote.get_branches().items()})
        self.assertEqual({'', 'blah', 'master'}, set(remote.branch_names()))

    def test_remove_tag(self):
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/tags/blah')
        remote = ControlDir.open(self.remote_url)
        remote_branch = remote.open_branch()
        remote_branch.tags.delete_tag('blah')
        self.assertRaises(NoSuchTag, remote_branch.tags.delete_tag, 'blah')
        self.assertEqual(self.remote_real.get_refs(), {b'refs/heads/master': self.remote_real.head(), b'HEAD': self.remote_real.head()})

    def test_set_tag(self):
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        remote = ControlDir.open(self.remote_url)
        remote.open_branch().tags.set_tag(b'blah', default_mapping.revision_id_foreign_to_bzr(c1))
        self.assertEqual(self.remote_real.get_refs(), {b'refs/heads/master': self.remote_real.head(), b'refs/tags/blah': c1, b'HEAD': self.remote_real.head()})

    def test_annotated_tag(self):
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        porcelain.tag_create(self.remote_real, tag=b'blah', author=b'author <author@example.com>', objectish=c2, tag_time=int(time.time()), tag_timezone=0, annotated=True, message=b'Annotated tag')
        remote = ControlDir.open(self.remote_url)
        remote_branch = remote.open_branch()
        self.assertEqual({'blah': default_mapping.revision_id_foreign_to_bzr(c2)}, remote_branch.tags.get_tag_dict())

    def test_get_branch_reference(self):
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        remote = ControlDir.open(self.remote_url)
        self.assertEqual(remote.user_url.rstrip('/') + ',branch=master', remote.get_branch_reference(''))
        self.assertEqual(None, remote.get_branch_reference('master'))

    def test_get_branch_nick(self):
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        remote = ControlDir.open(self.remote_url)
        self.assertEqual('master', remote.open_branch().nick)