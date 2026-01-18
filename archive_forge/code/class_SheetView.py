from datetime import datetime, time
from petl.compat import xrange
class SheetView(object):
    """
    A view on a sheet in a workbook. Should be created by indexing a
    :class:`View`.
    
    These can be sliced to create smaller views.
    
    Views can be iterated over to return a set of iterables, one for each row
    in the view. Data is returned as in the cell values with the exception of
    dates and times which are converted into :class:`~datetime.datetime`
    instances.
    """

    def __init__(self, book, sheet, row_slice=None, col_slice=None):
        self.book = book
        self.sheet = sheet
        for name, source in (('rows', row_slice), ('cols', col_slice)):
            start = 0
            stop = max_n = getattr(self.sheet, 'n' + name)
            if isinstance(source, slice):
                if source.start is not None:
                    start_val = source.start
                    if isinstance(start_val, Index):
                        start_val = start_val.__index__()
                    if start_val < 0:
                        start = max(0, max_n + start_val)
                    elif start_val > 0:
                        start = min(max_n, start_val)
                if source.stop is not None:
                    stop_val = source.stop
                    if isinstance(stop_val, Index):
                        stop_val = stop_val.__index__() + 1
                    if stop_val < 0:
                        stop = max(0, max_n + stop_val)
                    elif stop_val > 0:
                        stop = min(max_n, stop_val)
            setattr(self, name, xrange(start, stop))

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

    def __iter__(self):
        for rowx in self.rows:
            yield self.__row(rowx)

    def __getitem__(self, slices):
        assert isinstance(slices, tuple)
        assert len(slices) == 2
        return self.__class__(self.book, self.sheet, *slices)