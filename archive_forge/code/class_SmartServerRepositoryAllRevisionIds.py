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
class SmartServerRepositoryAllRevisionIds(SmartServerRepositoryRequest):
    """Retrieve all of the revision ids in a repository.

    New in 2.5.
    """

    def do_repository_request(self, repository):
        revids = repository.all_revision_ids()
        return SuccessfulSmartServerResponse((b'ok',), b'\n'.join(revids))