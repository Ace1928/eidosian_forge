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
class SmartServerRepositoryAddSignatureText(SmartServerRepositoryRequest):
    """Add a revision signature text.

    New in 2.5.
    """

    def do_repository_request(self, repository, lock_token, revision_id, *write_group_tokens):
        """Add a revision signature text.

        :param repository: Repository to operate on
        :param lock_token: Lock token
        :param revision_id: Revision for which to add signature
        :param write_group_tokens: Write group tokens
        """
        self._lock_token = lock_token
        self._revision_id = revision_id
        self._write_group_tokens = [token.decode('utf-8') for token in write_group_tokens]
        return None

    def do_body(self, body_bytes):
        """Add a signature text.

        :param body_bytes: GPG signature text
        :return: SuccessfulSmartServerResponse with arguments 'ok' and
            the list of new write group tokens.
        """
        with self._repository.lock_write(token=self._lock_token):
            self._repository.resume_write_group(self._write_group_tokens)
            try:
                self._repository.add_signature_text(self._revision_id, body_bytes)
            finally:
                new_write_group_tokens = self._repository.suspend_write_group()
        return SuccessfulSmartServerResponse((b'ok',) + tuple([token.encode('utf-8') for token in new_write_group_tokens]))