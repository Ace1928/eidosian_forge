from collections import namedtuple
def _decode_11000110_sssscccc(self):
    self._index += 1
    op1 = self._bytecode_array[self._index]
    self._index += 1
    start = (op1 & 240) >> 4
    count = (op1 & 15) >> 0
    return 'pop %s' % self._print_registers(self._calculate_range(start, count), 'wR')