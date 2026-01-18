import bz2
import os
import re
import sys
import zlib
from typing import Callable, List, Optional
import fastbencode as bencode
from .. import branch
from .. import bzr as _mod_bzr
from .. import config as _mod_config
from .. import (controldir, debug, errors, gpg, graph, lock, lockdir, osutils,
from .. import repository as _mod_repository
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..branch import BranchWriteLockResult
from ..decorators import only_raises
from ..errors import NoSuchRevision, SmartProtocolError
from ..i18n import gettext
from ..lockable_files import LockableFiles
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..revision import NULL_REVISION
from ..trace import log_exception_quietly, mutter, note, warning
from . import branch as bzrbranch
from . import bzrdir as _mod_bzrdir
from . import inventory_delta
from . import repository as bzrrepository
from . import testament as _mod_testament
from . import vf_repository, vf_search
from .branch import BranchReferenceFormat
from .inventory import Inventory
from .inventorytree import InventoryRevisionTree
from .serializer import format_registry as serializer_format_registry
from .smart import client
from .smart import repository as smart_repo
from .smart import vfs
from .smart.client import _SmartClient
from .versionedfile import FulltextContentFactory
def _annotate_file_revision(self, revid, tree_path, file_id, default_revision):
    path = self.controldir._path_for_remote_call(self._client)
    tree_path = tree_path.encode('utf-8')
    file_id = file_id or b''
    default_revision = default_revision or b''
    try:
        response, handler = self._call_expecting_body(b'Repository.annotate_file_revision', path, revid, tree_path, file_id, default_revision)
    except errors.UnknownSmartMethod:
        return None
    if response[0] != b'ok':
        raise errors.UnexpectedSmartServerResponse(response)
    return map(tuple, bencode.bdecode(handler.read_body_bytes()))