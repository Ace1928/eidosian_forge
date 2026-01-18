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
def _byte_stream_to_stream(byte_stream, record_counter=None):
    """Convert a byte stream into a format and a stream.

    :param byte_stream: A bytes iterator, as output by _stream_to_byte_stream.
    :return: (RepositoryFormat, stream_generator)
    """
    decoder = _ByteStreamDecoder(byte_stream, record_counter)
    for bytes in byte_stream:
        decoder.stream_decoder.accept_bytes(bytes)
        for record in decoder.stream_decoder.read_pending_records(max=1):
            record_names, src_format_name = record
            src_format = network_format_registry.get(src_format_name)
            return (src_format, decoder.record_stream())