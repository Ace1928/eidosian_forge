from collections import defaultdict
import param
from matplotlib.font_manager import FontProperties
from matplotlib.table import Table as mpl_Table
from .element import ElementPlot
from .plot import mpl_rc_context
def _render_table(self, element, axes):
    if self.dynamic:
        cell_widths = defaultdict(int)
        self._update_cell_widths(element, cell_widths)
    else:
        cell_widths = self.cell_widths
    size_factor = 1.0 - 2 * self.border
    table = mpl_Table(axes, bbox=[self.border, self.border, size_factor, size_factor])
    total_width = sum(cell_widths.values())
    height = size_factor / element.rows
    summarize = element.rows > self.max_rows
    half_rows = self.max_rows / 2
    rows = min([self.max_rows, element.rows])
    for row in range(rows):
        adjusted_row = row
        for col in range(element.cols):
            if summarize and row > half_rows:
                adjusted_row = element.rows - self.max_rows + row
            cell_value = self._cell_value(element, row, col)
            cellfont = self.font_types.get(element.cell_type(adjusted_row, col), None)
            width = cell_widths[col] / float(total_width)
            font_kwargs = dict(fontproperties=cellfont) if cellfont else {}
            table.add_cell(row, col, width, height, text=cell_value, loc='center', **font_kwargs)
    table.set_fontsize(self.max_font_size)
    table.auto_set_font_size(True)
    return table