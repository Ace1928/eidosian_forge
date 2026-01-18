import logging
from .table import HeaderTable, table_entry_size
from .compat import to_byte, to_bytes
from .exceptions import (
from .huffman import HuffmanEncoder
from .huffman_constants import (
from .huffman_table import decode_huffman
from .struct import HeaderTuple, NeverIndexedHeaderTuple
def _decode_literal(self, data, should_index):
    """
        Decodes a header represented with a literal.
        """
    total_consumed = 0
    if should_index:
        indexed_name = to_byte(data[0]) & 63
        name_len = 6
        not_indexable = False
    else:
        high_byte = to_byte(data[0])
        indexed_name = high_byte & 15
        name_len = 4
        not_indexable = high_byte & 16
    if indexed_name:
        index, consumed = decode_integer(data, name_len)
        name = self.header_table.get_by_index(index)[0]
        total_consumed = consumed
        length = 0
    else:
        data = data[1:]
        length, consumed = decode_integer(data, 7)
        name = data[consumed:consumed + length]
        if len(name) != length:
            raise HPACKDecodingError('Truncated header block')
        if to_byte(data[0]) & 128:
            name = decode_huffman(name)
        total_consumed = consumed + length + 1
    data = data[consumed + length:]
    length, consumed = decode_integer(data, 7)
    value = data[consumed:consumed + length]
    if len(value) != length:
        raise HPACKDecodingError('Truncated header block')
    if to_byte(data[0]) & 128:
        value = decode_huffman(value)
    total_consumed += length + consumed
    if not_indexable:
        header = NeverIndexedHeaderTuple(name, value)
    else:
        header = HeaderTuple(name, value)
    if should_index:
        self.header_table.add(name, value)
    log.debug('Decoded %s, total consumed %d bytes, indexed %s', header, total_consumed, should_index)
    return (header, total_consumed)