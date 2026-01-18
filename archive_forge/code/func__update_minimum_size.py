from kivy.logger import Logger
from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, BooleanProperty, DictProperty, \
from math import ceil
from itertools import accumulate, product, chain, islice
from operator import sub
def _update_minimum_size(self):
    l, t, r, b = self.padding
    spacing_x, spacing_y = self.spacing
    cols, rows = (self._cols, self._rows)
    width = l + r + spacing_x * (len(cols) - 1)
    self._cols_min_size_none = sum(cols) + width
    if self._has_hint_bound_x:
        cols_sh_min = self._cols_sh_min
        cols_sh_max = self._cols_sh_max
        for i, (c, sh_min, sh_max) in enumerate(zip(cols, cols_sh_min, cols_sh_max)):
            if sh_min is not None:
                width += max(c, sh_min)
                cols_sh_min[i] = max(0.0, sh_min - c)
            else:
                width += c
            if sh_max is not None:
                cols_sh_max[i] = max(0.0, sh_max - c)
    else:
        width = self._cols_min_size_none
    height = t + b + spacing_y * (len(rows) - 1)
    self._rows_min_size_none = sum(rows) + height
    if self._has_hint_bound_y:
        rows_sh_min = self._rows_sh_min
        rows_sh_max = self._rows_sh_max
        for i, (r, sh_min, sh_max) in enumerate(zip(rows, rows_sh_min, rows_sh_max)):
            if sh_min is not None:
                height += max(r, sh_min)
                rows_sh_min[i] = max(0.0, sh_min - r)
            else:
                height += r
            if sh_max is not None:
                rows_sh_max[i] = max(0.0, sh_max - r)
    else:
        height = self._rows_min_size_none
    self.minimum_size = (width, height)