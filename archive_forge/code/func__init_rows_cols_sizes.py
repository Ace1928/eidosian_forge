from kivy.logger import Logger
from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, BooleanProperty, DictProperty, \
from math import ceil
from itertools import accumulate, product, chain, islice
from operator import sub
def _init_rows_cols_sizes(self, count):
    current_cols = self.cols
    current_rows = self.rows
    if not current_cols and (not current_rows):
        Logger.warning('%r have no cols or rows set, layout is not triggered.' % self)
        return
    if current_cols is None:
        current_cols = int(ceil(count / float(current_rows)))
    elif current_rows is None:
        current_rows = int(ceil(count / float(current_cols)))
    current_cols = max(1, current_cols)
    current_rows = max(1, current_rows)
    self._has_hint_bound_x = False
    self._has_hint_bound_y = False
    self._cols_min_size_none = 0.0
    self._rows_min_size_none = 0.0
    self._cols = cols = [self.col_default_width] * current_cols
    self._cols_sh = [None] * current_cols
    self._cols_sh_min = [None] * current_cols
    self._cols_sh_max = [None] * current_cols
    self._rows = rows = [self.row_default_height] * current_rows
    self._rows_sh = [None] * current_rows
    self._rows_sh_min = [None] * current_rows
    self._rows_sh_max = [None] * current_rows
    items = (i for i in self.cols_minimum.items() if i[0] < len(cols))
    for index, value in items:
        cols[index] = max(value, cols[index])
    items = (i for i in self.rows_minimum.items() if i[0] < len(rows))
    for index, value in items:
        rows[index] = max(value, rows[index])
    return True