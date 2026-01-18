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
class SmartServerRepositoryStartWriteGroup(SmartServerRepositoryRequest):
    """Start a write group.

    New in 2.5.
    """

    def do_repository_request(self, repository, lock_token):
        """Start a write group."""
        with repository.lock_write(token=lock_token):
            repository.start_write_group()
            try:
                tokens = repository.suspend_write_group()
            except errors.UnsuspendableWriteGroup:
                return FailedSmartServerResponse((b'UnsuspendableWriteGroup',))
        return SuccessfulSmartServerResponse((b'ok', tokens))