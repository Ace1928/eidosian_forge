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
def _iter_files_bytes_rpc(self, desired_files, absent):
    path = self.controldir._path_for_remote_call(self._client)
    lines = []
    identifiers = []
    for file_id, revid, identifier in desired_files:
        lines.append(b''.join([file_id, b'\x00', revid]))
        identifiers.append(identifier)
    response_tuple, response_handler = self._call_with_body_bytes_expecting_body(b'Repository.iter_files_bytes', (path,), b'\n'.join(lines))
    if response_tuple != (b'ok',):
        response_handler.cancel_read_body()
        raise errors.UnexpectedSmartServerResponse(response_tuple)
    byte_stream = response_handler.read_streamed_body()

    def decompress_stream(start, byte_stream, unused):
        decompressor = zlib.decompressobj()
        yield decompressor.decompress(start)
        while decompressor.unused_data == b'':
            try:
                data = next(byte_stream)
            except StopIteration:
                break
            yield decompressor.decompress(data)
        yield decompressor.flush()
        unused.append(decompressor.unused_data)
    unused = b''
    while True:
        while b'\n' not in unused:
            try:
                unused += next(byte_stream)
            except StopIteration:
                return
        header, rest = unused.split(b'\n', 1)
        args = header.split(b'\x00')
        if args[0] == b'absent':
            absent[identifiers[int(args[3])]] = (args[1], args[2])
            unused = rest
            continue
        elif args[0] == b'ok':
            idx = int(args[1])
        else:
            raise errors.UnexpectedSmartServerResponse(args)
        unused_chunks = []
        yield (identifiers[idx], decompress_stream(rest, byte_stream, unused_chunks))
        unused = b''.join(unused_chunks)