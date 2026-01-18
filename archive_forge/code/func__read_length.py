from testtools.compat import _b
def _read_length(self):
    """Try to decode a length from the bytes."""
    count_chars = []
    for bytes in self.buffered_bytes:
        for pos in range(len(bytes)):
            byte = bytes[pos:pos + 1]
            if byte not in self._match_chars:
                break
            count_chars.append(byte)
            if byte == self._slash_n:
                break
    if not count_chars:
        return
    if count_chars[-1] != self._slash_n:
        return
    count_str = empty.join(count_chars)
    if self.strict:
        if count_str[-2:] != self._slash_rn:
            raise ValueError('chunk header invalid: %r' % count_str)
        if self._slash_r in count_str[:-2]:
            raise ValueError('too many CRs in chunk header %r' % count_str)
    self.body_length = int(count_str.rstrip(self._slash_nr), 16)
    excess_bytes = len(count_str)
    while excess_bytes:
        if excess_bytes >= len(self.buffered_bytes[0]):
            excess_bytes -= len(self.buffered_bytes[0])
            del self.buffered_bytes[0]
        else:
            self.buffered_bytes[0] = self.buffered_bytes[0][excess_bytes:]
            excess_bytes = 0
    if not self.body_length:
        self.state = self._finished
        if not self.buffered_bytes:
            return empty
    else:
        self.state = self._read_body
    return self.state()