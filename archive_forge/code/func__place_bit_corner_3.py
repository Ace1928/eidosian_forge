from reportlab.graphics.barcode.common import Barcode
def _place_bit_corner_3(self, data):
    bits = self._get_next_bits(data)
    self._place_bit(self.row_usable_modules - 3, 0, bits[0])
    self._place_bit(self.row_usable_modules - 2, 0, bits[1])
    self._place_bit(self.row_usable_modules - 1, 0, bits[2])
    self._place_bit(0, self.col_usable_modules - 2, bits[3])
    self._place_bit(0, self.col_usable_modules - 1, bits[4])
    self._place_bit(1, self.col_usable_modules - 1, bits[5])
    self._place_bit(2, self.col_usable_modules - 1, bits[6])
    self._place_bit(3, self.col_usable_modules - 1, bits[7])