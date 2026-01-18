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
class SmartServerRepositoryCommitWriteGroup(SmartServerRepositoryRequest):
    """Commit a write group.

    New in 2.5.
    """

    def do_repository_request(self, repository, lock_token, write_group_tokens):
        """Commit a write group."""
        with repository.lock_write(token=lock_token):
            try:
                repository.resume_write_group([token.decode('utf-8') for token in write_group_tokens])
            except errors.UnresumableWriteGroup as e:
                return FailedSmartServerResponse((b'UnresumableWriteGroup', [token.encode('utf-8') for token in e.write_groups], e.reason.encode('utf-8')))
            try:
                repository.commit_write_group()
            except:
                write_group_tokens = repository.suspend_write_group()
                raise
        return SuccessfulSmartServerResponse((b'ok',))