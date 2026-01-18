import logging
from .table import HeaderTable, table_entry_size
from .compat import to_byte, to_bytes
from .exceptions import (
from .huffman import HuffmanEncoder
from .huffman_constants import (
from .huffman_table import decode_huffman
from .struct import HeaderTuple, NeverIndexedHeaderTuple
def _assert_valid_table_size(self):
    """
        Check that the table size set by the encoder is lower than the maximum
        we expect to have.
        """
    if self.header_table_size > self.max_allowed_table_size:
        raise InvalidTableSizeError('Encoder did not shrink table size to within the max')