from datetime import datetime, time
from petl.compat import xrange
def __row(self, rowx):
    from xlrd import XL_CELL_DATE, xldate_as_tuple
    for colx in self.cols:
        value = self.sheet.cell_value(rowx, colx)
        if self.sheet.cell_type(rowx, colx) == XL_CELL_DATE:
            date_parts = xldate_as_tuple(value, self.book.datemode)
            if date_parts[0]:
                value = datetime(*date_parts)
            else:
                value = time(*date_parts[3:])
        yield value