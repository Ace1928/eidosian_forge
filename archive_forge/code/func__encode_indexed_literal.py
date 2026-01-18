import logging
from .table import HeaderTable, table_entry_size
from .compat import to_byte, to_bytes
from .exceptions import (
from .huffman import HuffmanEncoder
from .huffman_constants import (
from .huffman_table import decode_huffman
from .struct import HeaderTuple, NeverIndexedHeaderTuple
def _encode_indexed_literal(self, index, value, indexbit, huffman=False):
    """
        Encodes a header with an indexed name and a literal value and performs
        incremental indexing.
        """
    if indexbit != INDEX_INCREMENTAL:
        prefix = encode_integer(index, 4)
    else:
        prefix = encode_integer(index, 6)
    prefix[0] |= ord(indexbit)
    if huffman:
        value = self.huffman_coder.encode(value)
    value_len = encode_integer(len(value), 7)
    if huffman:
        value_len[0] |= 128
    return b''.join([bytes(prefix), bytes(value_len), value])