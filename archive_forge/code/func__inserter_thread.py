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
def _inserter_thread(self):
    try:
        src_format, stream = _byte_stream_to_stream(self.blocking_byte_stream())
        self.insert_result = self.repository._get_sink().insert_stream(stream, src_format, self.tokens)
        self.insert_ok = True
    except:
        self.insert_exception = sys.exc_info()
        self.insert_ok = False