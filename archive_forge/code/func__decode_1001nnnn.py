from collections import namedtuple
def _decode_1001nnnn(self):
    opcode = self._bytecode_array[self._index]
    self._index += 1
    return 'vsp = r%u' % (opcode & 15)