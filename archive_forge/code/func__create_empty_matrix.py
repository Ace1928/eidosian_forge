from reportlab.graphics.barcode.common import Barcode
def _create_empty_matrix(self, row, col):
    matrix = []
    for i in range(0, row):
        matrix.append([None] * col)
    return matrix