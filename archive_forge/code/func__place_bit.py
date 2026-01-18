from reportlab.graphics.barcode.common import Barcode
def _place_bit(self, row, col, bit):
    if row < 0:
        row += self.row_usable_modules
        col += 4 - (self.row_usable_modules + 4) % 8
    if col < 0:
        col += self.col_usable_modules
        row += 4 - (self.col_usable_modules + 4) % 8
    self._matrix[row][col] = bit