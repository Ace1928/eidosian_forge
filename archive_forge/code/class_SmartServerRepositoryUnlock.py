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
class SmartServerRepositoryUnlock(SmartServerRepositoryRequest):

    def do_repository_request(self, repository, token):
        try:
            repository.lock_write(token=token)
        except errors.TokenMismatch as e:
            return FailedSmartServerResponse((b'TokenMismatch',))
        repository.dont_leave_lock_in_place()
        repository.unlock()
        return SuccessfulSmartServerResponse((b'ok',))