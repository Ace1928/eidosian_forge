import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _read_prefix(self, stream):
    signature = stream.read(len(self._signature()))
    if not signature == self._signature():
        raise BadIndexFormatSignature(self._name, GraphIndex)
    options_line = stream.readline()
    if not options_line.startswith(_OPTION_NODE_REFS):
        raise BadIndexOptions(self)
    try:
        self.node_ref_lists = int(options_line[len(_OPTION_NODE_REFS):-1])
    except ValueError:
        raise BadIndexOptions(self)
    options_line = stream.readline()
    if not options_line.startswith(_OPTION_KEY_ELEMENTS):
        raise BadIndexOptions(self)
    try:
        self._key_length = int(options_line[len(_OPTION_KEY_ELEMENTS):-1])
    except ValueError:
        raise BadIndexOptions(self)
    options_line = stream.readline()
    if not options_line.startswith(_OPTION_LEN):
        raise BadIndexOptions(self)
    try:
        self._key_count = int(options_line[len(_OPTION_LEN):-1])
    except ValueError:
        raise BadIndexOptions(self)