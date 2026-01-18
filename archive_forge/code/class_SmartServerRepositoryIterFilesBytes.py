import bz2
import itertools
import os
import queue
import sys
import tempfile
import threading
import zlib
import fastbencode as bencode
from ... import errors, estimate_compressed_size, osutils
from ... import revision as _mod_revision
from ... import trace, ui
from ...repository import _strip_NULL_ghosts, network_format_registry
from .. import inventory as _mod_inventory
from .. import inventory_delta, pack, vf_search
from ..bzrdir import BzrDir
from ..versionedfile import (ChunkedContentFactory, NetworkRecordStream,
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRepositoryIterFilesBytes(SmartServerRepositoryRequest):
    """Iterate over the contents of files.

    The client sends a list of desired files to stream, one
    per line, and as tuples of file id and revision, separated by
    \x00.

    The server replies with a stream. Each entry is preceded by a header,
    which can either be:

    * "ok\x00IDX
" where IDX is the index of the entry in the desired files
        list sent by the client. This header is followed by the contents of
        the file, bzip2-compressed.
    * "absent\x00FILEID\x00REVISION\x00IDX" to indicate a text is missing.
        The client can then raise an appropriate RevisionNotPresent error
        or check its fallback repositories.

    New in 2.5.
    """

    def body_stream(self, repository, desired_files):
        with self._repository.lock_read():
            text_keys = {}
            for i, key in enumerate(desired_files):
                text_keys[key] = i
            for record in repository.texts.get_record_stream(text_keys, 'unordered', True):
                identifier = text_keys[record.key]
                if record.storage_kind == 'absent':
                    yield (b'absent\x00%s\x00%s\x00%d\n' % (record.key[0], record.key[1], identifier))
                    continue
                yield (b'ok\x00%d\n' % identifier)
                compressor = zlib.compressobj()
                for bytes in record.iter_bytes_as('chunked'):
                    data = compressor.compress(bytes)
                    if data:
                        yield data
                data = compressor.flush()
                if data:
                    yield data

    def do_body(self, body_bytes):
        desired_files = [tuple(l.split(b'\x00')) for l in body_bytes.splitlines()]
        return SuccessfulSmartServerResponse((b'ok',), body_stream=self.body_stream(self._repository, desired_files))

    def do_repository_request(self, repository):
        return None