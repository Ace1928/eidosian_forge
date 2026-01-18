import logging
from .table import HeaderTable, table_entry_size
from .compat import to_byte, to_bytes
from .exceptions import (
from .huffman import HuffmanEncoder
from .huffman_constants import (
from .huffman_table import decode_huffman
from .struct import HeaderTuple, NeverIndexedHeaderTuple
def _decode_indexed(self, data):
    """
        Decodes a header represented using the indexed representation.
        """
    index, consumed = decode_integer(data, 7)
    header = HeaderTuple(*self.header_table.get_by_index(index))
    log.debug('Decoded %s, consumed %d', header, consumed)
    return (header, consumed)