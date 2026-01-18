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
class SmartServerRepositoryReconcile(SmartServerRepositoryRequest):
    """Reconcile a repository.

    New in 2.5.
    """

    def do_repository_request(self, repository, lock_token):
        try:
            repository.lock_write(token=lock_token)
        except errors.TokenLockingNotSupported as e:
            return FailedSmartServerResponse((b'TokenLockingNotSupported',))
        try:
            reconciler = repository.reconcile()
        finally:
            repository.unlock()
        body = [b'garbage_inventories: %d\n' % reconciler.garbage_inventories, b'inconsistent_parents: %d\n' % reconciler.inconsistent_parents]
        return SuccessfulSmartServerResponse((b'ok',), b''.join(body))