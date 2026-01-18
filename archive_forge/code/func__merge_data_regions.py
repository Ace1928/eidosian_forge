from reportlab.graphics.barcode.common import Barcode
def _merge_data_regions(self, regions):
    merged = []
    for i in range(0, len(regions), self.row_regions):
        chunk = regions[i:i + self.row_regions]
        j = 0
        while j < len(chunk[0]):
            merged_row = []
            for row in chunk:
                merged_row += row[j]
            merged.append(merged_row)
            j += 1
    return merged