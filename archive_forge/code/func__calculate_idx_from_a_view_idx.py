import itertools
from kivy.uix.recyclelayout import RecycleLayout
from kivy.uix.gridlayout import GridLayout, GridLayoutException, nmax, nmin
from collections import defaultdict
def _calculate_idx_from_a_view_idx(self, n_cols, n_rows, view_idx):
    """returns a tuple of (column-index, row-index) from a view-index"""
    if self._fills_row_first:
        row_idx, col_idx = divmod(view_idx, n_cols)
    else:
        col_idx, row_idx = divmod(view_idx, n_rows)
    if not self._fills_from_left_to_right:
        col_idx = n_cols - col_idx - 1
    if not self._fills_from_top_to_bottom:
        row_idx = n_rows - row_idx - 1
    return (col_idx, row_idx)