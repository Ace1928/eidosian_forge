from kivy.logger import Logger
from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, BooleanProperty, DictProperty, \
from math import ceil
from itertools import accumulate, product, chain, islice
from operator import sub
def _iterate_layout(self, count):
    orientation = self.orientation
    padding = self.padding
    spacing_x, spacing_y = self.spacing
    cols = self._cols
    if self._fills_from_left_to_right:
        x_iter = accumulate(chain((self.x + padding[0],), (col_width + spacing_x for col_width in islice(cols, len(cols) - 1))))
    else:
        x_iter = accumulate(chain((self.right - padding[2] - cols[-1],), (col_width + spacing_x for col_width in islice(reversed(cols), 1, None))), sub)
        cols = reversed(cols)
    rows = self._rows
    if self._fills_from_top_to_bottom:
        y_iter = accumulate(chain((self.top - padding[1] - rows[0],), (row_height + spacing_y for row_height in islice(rows, 1, None))), sub)
    else:
        y_iter = accumulate(chain((self.y + padding[3],), (row_height + spacing_y for row_height in islice(reversed(rows), len(rows) - 1))))
        rows = reversed(rows)
    if self._fills_row_first:
        for i, (y, x), (row_height, col_width) in zip(reversed(range(count)), product(y_iter, x_iter), product(rows, cols)):
            yield (i, x, y, col_width, row_height)
    else:
        for i, (x, y), (col_width, row_height) in zip(reversed(range(count)), product(x_iter, y_iter), product(cols, rows)):
            yield (i, x, y, col_width, row_height)