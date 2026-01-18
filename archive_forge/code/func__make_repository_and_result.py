import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from breezy import branch as _mod_branch
from breezy import controldir, errors, gpg, tests, transport, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import inventory_delta, versionedfile
from breezy.bzr.smart import branch as smart_branch
from breezy.bzr.smart import bzrdir as smart_dir
from breezy.bzr.smart import packrepository as smart_packrepo
from breezy.bzr.smart import repository as smart_repo
from breezy.bzr.smart import request as smart_req
from breezy.bzr.smart import server, vfs
from breezy.bzr.testament import Testament
from breezy.tests import test_server
from breezy.transport import chroot, memory
def _make_repository_and_result(self, shared=False, format=None):
    """Convenience function to setup a repository.

        :result: The SmartServerResponse to expect when opening it.
        """
    repo = self.make_repository('.', shared=shared, format=format)
    if repo.supports_rich_root():
        rich_root = b'yes'
    else:
        rich_root = b'no'
    if repo._format.supports_tree_reference:
        subtrees = b'yes'
    else:
        subtrees = b'no'
    if repo._format.supports_external_lookups:
        external = b'yes'
    else:
        external = b'no'
    if smart_dir.SmartServerRequestFindRepositoryV3 == self._request_class:
        return smart_req.SuccessfulSmartServerResponse((b'ok', b'', rich_root, subtrees, external, repo._format.network_name()))
    elif smart_dir.SmartServerRequestFindRepositoryV2 == self._request_class:
        return smart_req.SuccessfulSmartServerResponse((b'ok', b'', rich_root, subtrees, external))
    else:
        return smart_req.SuccessfulSmartServerResponse((b'ok', b'', rich_root, subtrees))