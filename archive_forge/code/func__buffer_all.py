import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _buffer_all(self, stream=None):
    """Buffer all the index data.

        Mutates self._nodes and self.keys_by_offset.
        """
    if self._nodes is not None:
        return
    if 'index' in debug.debug_flags:
        trace.mutter('Reading entire index %s', self._transport.abspath(self._name))
    if stream is None:
        stream = self._transport.get(self._name)
        if self._base_offset != 0:
            stream = BytesIO(stream.read()[self._base_offset:])
    try:
        self._read_prefix(stream)
        self._expected_elements = 3 + self._key_length
        line_count = 0
        self._keys_by_offset = {}
        self._nodes = {}
        self._nodes_by_key = None
        trailers = 0
        pos = stream.tell()
        lines = stream.read().split(b'\n')
    finally:
        stream.close()
    del lines[-1]
    _, _, _, trailers = self._parse_lines(lines, pos)
    for key, absent, references, value in self._keys_by_offset.values():
        if absent:
            continue
        if self.node_ref_lists:
            node_value = (value, self._resolve_references(references))
        else:
            node_value = value
        self._nodes[key] = node_value
    if trailers != 1:
        raise BadIndexData(self)