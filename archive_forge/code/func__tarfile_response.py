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
def _tarfile_response(self, tmp_dirname, compression):
    with tempfile.NamedTemporaryFile() as temp:
        self._tarball_of_dir(tmp_dirname, compression, temp.file)
        temp.seek(0)
        return SuccessfulSmartServerResponse((b'ok',), temp.read())