from collections import namedtuple
def _decode_01xxxxxx(self):
    opcode = self._bytecode_array[self._index]
    self._index += 1
    return 'vsp = vsp - %u' % (((opcode & 63) << 2) + 4)