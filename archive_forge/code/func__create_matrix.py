from reportlab.graphics.barcode.common import Barcode
def _create_matrix(self, data):
    """
        This method is heavily influenced by "huBarcode" which is BSD licensed
        https://github.com/hudora/huBarcode/blob/master/hubarcode/datamatrix/placement.py
        """
    rows = self.row_usable_modules
    cols = self.col_usable_modules
    self._matrix = self._create_empty_matrix(rows, cols)
    row = 4
    col = 0
    while True:
        if row == rows and col == 0:
            self._place_bit_corner_1(data)
        elif row == rows - 2 and col == 0 and cols % 4:
            self._place_bit_corner_2(data)
        elif row == rows - 2 and col == 0 and (cols % 8 == 4):
            self._place_bit_corner_3(data)
        elif row == rows + 4 and col == 2 and (cols % 8 == 0):
            self._place_bit_corner_4(data)
        while True:
            if row < rows and col >= 0 and (self._matrix[row][col] is None):
                self._place_bit_standard(data, row, col)
            row -= 2
            col += 2
            if row < 0 or col >= cols:
                break
        row += 1
        col += 3
        while True:
            if row >= 0 and col < cols and (self._matrix[row][col] is None):
                self._place_bit_standard(data, row, col)
            row += 2
            col -= 2
            if row >= rows or col < 0:
                break
        row += 3
        col += 1
        if row >= rows and col >= cols:
            break
    for row in self._matrix:
        for i in range(0, cols):
            if row[i] is None:
                row[i] = 0
    return self._matrix