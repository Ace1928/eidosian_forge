from reportlab.graphics.barcode.common import Barcode
def _wrap_data_regions_with_finders(self, regions):
    wrapped = []
    for region in regions:
        matrix = self._create_empty_matrix(int(self.col_modules / self.col_regions), int(self.row_modules / self.row_regions))
        for i, rows in enumerate(region):
            for j, data in enumerate(rows):
                matrix[i + 1][j + 1] = data
        for i, row in enumerate(matrix):
            if i == 0:
                for j, col in enumerate(row):
                    row[j] = (j + 1) % 2
            elif i + 1 == len(matrix):
                for j, col in enumerate(row):
                    row[j] = 1
            else:
                row[0] = 1
                row[-1] = i % 2
        wrapped.append(matrix)
    return wrapped