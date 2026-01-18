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
class SmartServerRepositoryPack(SmartServerRepositoryRequest):
    """Pack a repository.

    New in 2.5.
    """

    def do_repository_request(self, repository, lock_token, clean_obsolete_packs):
        self._repository = repository
        self._lock_token = lock_token
        if clean_obsolete_packs == b'True':
            self._clean_obsolete_packs = True
        else:
            self._clean_obsolete_packs = False
        return None

    def do_body(self, body_bytes):
        if body_bytes == '':
            hint = None
        else:
            hint = body_bytes.splitlines()
        with self._repository.lock_write(token=self._lock_token):
            self._repository.pack(hint, self._clean_obsolete_packs)
        return SuccessfulSmartServerResponse((b'ok',))