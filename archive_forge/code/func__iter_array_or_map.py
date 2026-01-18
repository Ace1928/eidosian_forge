from struct import unpack
def _iter_array_or_map(self):
    """Each block consists of a long count value, followed by that many
        array items. A block with count zero indicates the end of the array.
        Each item is encoded per the array's item schema.

        If a block's count is negative, then the count is followed immediately
        by a long block size, indicating the number of bytes in the block.
        The actual count in this case is the absolute value of the count
        written.
        """
    while self._block_count != 0:
        if self._block_count < 0:
            self._block_count = -self._block_count
            self.read_long()
        for i in range(self._block_count):
            yield
        self._block_count = self.read_long()