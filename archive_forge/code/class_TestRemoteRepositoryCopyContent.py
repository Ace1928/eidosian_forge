import base64
import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from ... import branch, config, controldir, errors, repository, tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...branch import Branch
from ...revision import NULL_REVISION, Revision
from ...tests import test_server
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from ...transport.remote import (RemoteSSHTransport, RemoteTCPTransport,
from .. import (RemoteBzrProber, bzrdir, groupcompress_repo, inventory,
from ..bzrdir import BzrDir, BzrDirFormat
from ..chk_serializer import chk_bencode_serializer
from ..remote import (RemoteBranch, RemoteBranchFormat, RemoteBzrDir,
from ..smart import medium, request
from ..smart.client import _SmartClient
from ..smart.repository import (SmartServerRepositoryGetParentMap,
class TestRemoteRepositoryCopyContent(tests.TestCaseWithTransport):
    """RemoteRepository.copy_content_into optimizations"""

    def test_copy_content_remote_to_local(self):
        self.transport_server = test_server.SmartTCPServer_for_testing
        src_repo = self.make_repository('repo1')
        src_repo = repository.Repository.open(self.get_url('repo1'))
        dest_url = self.get_vfs_only_url('repo2')
        dest_bzrdir = BzrDir.create(dest_url)
        dest_repo = dest_bzrdir.create_repository()
        self.assertFalse(isinstance(dest_repo, RemoteRepository))
        self.assertTrue(isinstance(src_repo, RemoteRepository))
        src_repo.copy_content_into(dest_repo)