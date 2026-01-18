from collections import namedtuple
def _decode_11000nnn(self):
    opcode = self._bytecode_array[self._index]
    self._index += 1
    return 'pop %s' % self._print_registers(self._calculate_range(10, opcode & 7), 'wR')