import logging
from .table import HeaderTable, table_entry_size
from .compat import to_byte, to_bytes
from .exceptions import (
from .huffman import HuffmanEncoder
from .huffman_constants import (
from .huffman_table import decode_huffman
from .struct import HeaderTuple, NeverIndexedHeaderTuple
def _encode_indexed(self, index):
    """
        Encodes a header using the indexed representation.
        """
    field = encode_integer(index, 7)
    field[0] |= 128
    return bytes(field)