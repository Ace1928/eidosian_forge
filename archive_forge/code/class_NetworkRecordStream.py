import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
class NetworkRecordStream:
    """A record_stream which reconstitures a serialised stream."""

    def __init__(self, bytes_iterator):
        """Create a NetworkRecordStream.

        :param bytes_iterator: An iterator of bytes. Each item in this
            iterator should have been obtained from a record_streams'
            record.get_bytes_as(record.storage_kind) call.
        """
        self._bytes_iterator = bytes_iterator
        self._kind_factory = {'fulltext': fulltext_network_to_record, 'groupcompress-block': groupcompress.network_block_to_records, 'knit-ft-gz': knit.knit_network_to_record, 'knit-delta-gz': knit.knit_network_to_record, 'knit-annotated-ft-gz': knit.knit_network_to_record, 'knit-annotated-delta-gz': knit.knit_network_to_record, 'knit-delta-closure': knit.knit_delta_closure_to_records}

    def read(self):
        """Read the stream.

        :return: An iterator as per VersionedFiles.get_record_stream().
        """
        for bytes in self._bytes_iterator:
            storage_kind, line_end = network_bytes_to_kind_and_offset(bytes)
            yield from self._kind_factory[storage_kind](storage_kind, bytes, line_end)