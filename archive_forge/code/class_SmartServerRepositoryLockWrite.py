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
class SmartServerRepositoryLockWrite(SmartServerRepositoryRequest):

    def do_repository_request(self, repository, token=b''):
        if token == b'':
            token = None
        try:
            token = repository.lock_write(token=token).repository_token
        except errors.LockContention as e:
            return FailedSmartServerResponse((b'LockContention',))
        except errors.UnlockableTransport:
            return FailedSmartServerResponse((b'UnlockableTransport',))
        except errors.LockFailed as e:
            return FailedSmartServerResponse((b'LockFailed', str(e.lock), str(e.why)))
        if token is not None:
            repository.leave_lock_in_place()
        repository.unlock()
        if token is None:
            token = b''
        return SuccessfulSmartServerResponse((b'ok', token))