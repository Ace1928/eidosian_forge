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
class SmartServerRepositoryGetStream(SmartServerRepositoryRequest):

    def do_repository_request(self, repository, to_network_name):
        """Get a stream for inserting into a to_format repository.

        The request body is 'search_bytes', a description of the revisions
        being requested.

        In 2.3 this verb added support for search_bytes == 'everything'.  Older
        implementations will respond with a BadSearch error, and clients should
        catch this and fallback appropriately.

        :param repository: The repository to stream from.
        :param to_network_name: The network name of the format of the target
            repository.
        """
        self._to_format = network_format_registry.get(to_network_name)
        if self._should_fake_unknown():
            return FailedSmartServerResponse((b'UnknownMethod', b'Repository.get_stream'))
        return None

    def _should_fake_unknown(self):
        """Return True if we should return UnknownMethod to the client.

        This is a workaround for bugs in pre-1.19 clients that claim to
        support receiving streams of CHK repositories.  The pre-1.19 client
        expects inventory records to be serialized in the format defined by
        to_network_name, but in pre-1.19 (at least) that format definition
        tries to use the xml5 serializer, which does not correctly handle
        rich-roots.  After 1.19 the client can also accept inventory-deltas
        (which avoids this issue), and those clients will use the
        Repository.get_stream_1.19 verb instead of this one.
        So: if this repository is CHK, and the to_format doesn't match,
        we should just fake an UnknownSmartMethod error so that the client
        will fallback to VFS, rather than sending it a stream we know it
        cannot handle.
        """
        from_format = self._repository._format
        to_format = self._to_format
        if not from_format.supports_chks:
            return False
        if to_format.supports_chks and from_format.repository_class is to_format.repository_class and (from_format._serializer == to_format._serializer):
            return False
        return True

    def do_body(self, body_bytes):
        repository = self._repository
        repository.lock_read()
        try:
            search_result, error = self.recreate_search(repository, body_bytes, discard_excess=True)
            if error is not None:
                repository.unlock()
                return error
            source = repository._get_source(self._to_format)
            stream = source.get_stream(search_result)
        except Exception:
            try:
                repository.unlock()
            finally:
                raise
        return SuccessfulSmartServerResponse((b'ok',), body_stream=self.body_stream(stream, repository))

    def body_stream(self, stream, repository):
        byte_stream = _stream_to_byte_stream(stream, repository._format)
        try:
            yield from byte_stream
        except errors.RevisionNotPresent as e:
            repository.unlock()
            yield FailedSmartServerResponse((b'NoSuchRevision', e.revision_id))
        else:
            repository.unlock()