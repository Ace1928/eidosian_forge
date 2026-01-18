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
class RemoteStreamSink(vf_repository.StreamSink):

    def _insert_real(self, stream, src_format, resume_tokens):
        self.target_repo._ensure_real()
        sink = self.target_repo._real_repository._get_sink()
        result = sink.insert_stream(stream, src_format, resume_tokens)
        if not result:
            self.target_repo.autopack()
        return result

    def insert_missing_keys(self, source, missing_keys):
        if isinstance(source, RemoteStreamSource) and source.from_repository._client._medium == self.target_repo._client._medium:
            stream = source._get_real_stream_for_missing_keys(missing_keys)
        else:
            stream = source.get_stream_for_missing_keys(missing_keys)
        return self.insert_stream_without_locking(stream, self.target_repo._format)

    def insert_stream(self, stream, src_format, resume_tokens):
        target = self.target_repo
        target._unstacked_provider.missing_keys.clear()
        candidate_calls = [(b'Repository.insert_stream_1.19', (1, 19))]
        if target._lock_token:
            candidate_calls.append((b'Repository.insert_stream_locked', (1, 14)))
            lock_args = (target._lock_token or b'',)
        else:
            candidate_calls.append((b'Repository.insert_stream', (1, 13)))
            lock_args = ()
        client = target._client
        medium = client._medium
        path = target.controldir._path_for_remote_call(client)
        found_verb = False
        for verb, required_version in candidate_calls:
            if medium._is_remote_before(required_version):
                continue
            if resume_tokens:
                found_verb = True
                break
            byte_stream = smart_repo._stream_to_byte_stream([], src_format)
            try:
                response = client.call_with_body_stream((verb, path, b'') + lock_args, byte_stream)
            except errors.UnknownSmartMethod:
                medium._remember_remote_is_before(required_version)
            else:
                found_verb = True
                break
        if not found_verb:
            return self._insert_real(stream, src_format, resume_tokens)
        self._last_inv_record = None
        self._last_substream = None
        if required_version < (1, 19):
            stream = self._stop_stream_if_inventory_delta(stream)
        byte_stream = smart_repo._stream_to_byte_stream(stream, src_format)
        resume_tokens = b' '.join([token.encode('utf-8') for token in resume_tokens])
        response = client.call_with_body_stream((verb, path, resume_tokens) + lock_args, byte_stream)
        if response[0][0] not in (b'ok', b'missing-basis'):
            raise errors.UnexpectedSmartServerResponse(response)
        if self._last_substream is not None:
            self.target_repo.refresh_data()
            return self._resume_stream_with_vfs(response, src_format)
        if response[0][0] == b'missing-basis':
            tokens, missing_keys = bencode.bdecode_as_tuple(response[0][1])
            resume_tokens = [token.decode('utf-8') for token in tokens]
            return (resume_tokens, {(entry[0].decode('utf-8'),) + entry[1:] for entry in missing_keys})
        else:
            self.target_repo.refresh_data()
            return ([], set())

    def _resume_stream_with_vfs(self, response, src_format):
        """Resume sending a stream via VFS, first resending the record and
        substream that couldn't be sent via an insert_stream verb.
        """
        if response[0][0] == b'missing-basis':
            tokens, missing_keys = bencode.bdecode_as_tuple(response[0][1])
            tokens = [token.decode('utf-8') for token in tokens]
        else:
            tokens = []

        def resume_substream():
            yield from self._last_substream
            self._last_substream = None

        def resume_stream():
            yield ('inventory-deltas', resume_substream())
            yield from self._last_stream
        return self._insert_real(resume_stream(), src_format, tokens)

    def _stop_stream_if_inventory_delta(self, stream):
        """Normally this just lets the original stream pass-through unchanged.

        However if any 'inventory-deltas' substream occurs it will stop
        streaming, and store the interrupted substream and stream in
        self._last_substream and self._last_stream so that the stream can be
        resumed by _resume_stream_with_vfs.
        """
        stream_iter = iter(stream)
        for substream_kind, substream in stream_iter:
            if substream_kind == 'inventory-deltas':
                self._last_substream = substream
                self._last_stream = stream_iter
                return
            else:
                yield (substream_kind, substream)