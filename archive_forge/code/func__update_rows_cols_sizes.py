import itertools
from kivy.uix.recyclelayout import RecycleLayout
from kivy.uix.gridlayout import GridLayout, GridLayoutException, nmax, nmin
from collections import defaultdict
def _update_rows_cols_sizes(self, changed):
    cols_count, rows_count = (self._cols_count, self._rows_count)
    cols, rows = (self._cols, self._rows)
    remove_view = self.remove_view
    n_cols = len(cols)
    n_rows = len(rows)
    orientation = self.orientation
    for index, widget, (w, h), (wn, hn), sh, shn, sh_min, shn_min, sh_max, shn_max, _, _ in changed:
        if sh != shn or sh_min != shn_min or sh_max != shn_max:
            return True
        elif sh[0] is not None and w != wn and (h == hn or sh[1] is not None) or (sh[1] is not None and h != hn and (w == wn or sh[0] is not None)):
            remove_view(widget, index)
        else:
            col, row = self._calculate_idx_from_a_view_idx(n_cols, n_rows, index)
            if w != wn:
                col_w = cols[col]
                cols_count[col][w] -= 1
                cols_count[col][wn] += 1
                was_last_w = cols_count[col][w] <= 0
                if was_last_w and col_w == w or wn > col_w:
                    return True
                if was_last_w:
                    del cols_count[col][w]
            if h != hn:
                row_h = rows[row]
                rows_count[row][h] -= 1
                rows_count[row][hn] += 1
                was_last_h = rows_count[row][h] <= 0
                if was_last_h and row_h == h or hn > row_h:
                    return True
                if was_last_h:
                    del rows_count[row][h]
    return False