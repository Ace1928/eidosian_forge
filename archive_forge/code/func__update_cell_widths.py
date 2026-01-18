from collections import defaultdict
import param
from matplotlib.font_manager import FontProperties
from matplotlib.table import Table as mpl_Table
from .element import ElementPlot
from .plot import mpl_rc_context
def _update_cell_widths(self, element, cell_widths):
    summarize = element.rows > self.max_rows
    half_rows = self.max_rows // 2
    rows = min([self.max_rows, element.rows])
    for row in range(rows):
        adjusted_row = row
        for col in range(element.cols):
            if summarize and row == half_rows:
                cell_text = '...'
            else:
                if summarize and row > half_rows:
                    adjusted_row = element.rows - self.max_rows + row
                cell_text = element.pprint_cell(adjusted_row, col)
                if len(cell_text) > self.max_value_len:
                    cell_text = cell_text[:self.max_value_len - 3] + '...'
            if len(cell_text) + 2 > cell_widths[col]:
                cell_widths[col] = len(cell_text) + 2