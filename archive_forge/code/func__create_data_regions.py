from reportlab.graphics.barcode.common import Barcode
def _create_data_regions(self, matrix):
    regions = []
    col_offset = 0
    row_offset = 0
    rows = int(self.row_usable_modules / self.row_regions)
    cols = int(self.col_usable_modules / self.col_regions)
    while col_offset < self.row_regions:
        while row_offset < self.col_regions:
            r_offset = col_offset * rows
            c_offset = row_offset * cols
            region = matrix[r_offset:rows + r_offset]
            for i in range(0, len(region)):
                region[i] = region[i][c_offset:cols + c_offset]
            regions.append(region)
            row_offset += 1
        row_offset = 0
        col_offset += 1
    return regions