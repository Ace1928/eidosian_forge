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
def _iter_inventories_rpc(self, revision_ids, ordering):
    if ordering is None:
        ordering = 'unordered'
    path = self.controldir._path_for_remote_call(self._client)
    body = b'\n'.join(revision_ids)
    response_tuple, response_handler = self._call_with_body_bytes_expecting_body(b'VersionedFileRepository.get_inventories', (path, ordering.encode('ascii')), body)
    if response_tuple[0] != b'ok':
        raise errors.UnexpectedSmartServerResponse(response_tuple)
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    byte_stream = response_handler.read_streamed_body()
    decoded = smart_repo._byte_stream_to_stream(byte_stream)
    if decoded is None:
        return
    src_format, stream = decoded
    if src_format.network_name() != self._format.network_name():
        raise AssertionError('Mismatched RemoteRepository and stream src {!r}, {!r}'.format(src_format.network_name(), self._format.network_name()))
    prev_inv = Inventory(root_id=None, revision_id=_mod_revision.NULL_REVISION)
    try:
        substream_kind, substream = next(stream)
    except StopIteration:
        return
    if substream_kind != 'inventory-deltas':
        raise AssertionError('Unexpected stream %r received' % substream_kind)
    for record in substream:
        parent_id, new_id, versioned_root, tree_references, invdelta = deserializer.parse_text_bytes(record.get_bytes_as('lines'))
        if parent_id != prev_inv.revision_id:
            raise AssertionError('invalid base {!r} != {!r}'.format(parent_id, prev_inv.revision_id))
        inv = prev_inv.create_by_apply_delta(invdelta, new_id)
        yield (inv, inv.revision_id)
        prev_inv = inv