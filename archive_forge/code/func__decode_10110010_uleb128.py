from collections import namedtuple
def _decode_10110010_uleb128(self):
    self._index += 1
    uleb_buffer = [self._bytecode_array[self._index]]
    self._index += 1
    while self._bytecode_array[self._index] & 128 == 0:
        uleb_buffer.append(self._bytecode_array[self._index])
        self._index += 1
    value = 0
    for b in reversed(uleb_buffer):
        value = (value << 7) + (b & 127)
    return 'vsp = vsp + %u' % (516 + (value << 2))