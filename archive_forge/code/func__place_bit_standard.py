from reportlab.graphics.barcode.common import Barcode
def _place_bit_standard(self, data, row, col):
    bits = self._get_next_bits(data)
    self._place_bit(row - 2, col - 2, bits[0])
    self._place_bit(row - 2, col - 1, bits[1])
    self._place_bit(row - 1, col - 2, bits[2])
    self._place_bit(row - 1, col - 1, bits[3])
    self._place_bit(row - 1, col, bits[4])
    self._place_bit(row, col - 2, bits[5])
    self._place_bit(row, col - 1, bits[6])
    self._place_bit(row, col, bits[7])