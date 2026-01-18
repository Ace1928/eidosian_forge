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
def _get_next_response(self):
    try:
        response_tuple = self.responses.pop(0)
    except IndexError:
        raise AssertionError("{!r} didn't expect any more calls".format(self))
    if response_tuple[0] == b'unknown':
        raise errors.UnknownSmartMethod(response_tuple[1])
    elif response_tuple[0] == b'error':
        raise errors.ErrorFromSmartServer(response_tuple[1])
    return response_tuple