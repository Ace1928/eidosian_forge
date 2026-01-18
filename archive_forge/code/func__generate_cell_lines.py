import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
def _generate_cell_lines(self, cell_data: Any, is_header: bool, col: Column, fill_char: str) -> Tuple[Deque[str], int]:
    """
        Generate the lines of a table cell

        :param cell_data: data to be included in cell
        :param is_header: True if writing a header cell, otherwise writing a data cell. This determines whether to
                          use header or data alignment settings as well as maximum lines to wrap.
        :param col: Column definition for this cell
        :param fill_char: character that fills remaining space in a cell. If your text has a background color,
                          then give fill_char the same background color. (Cannot be a line breaking character)
        :return: Tuple(deque of cell lines, display width of the cell)
        """
    data_str = str(cell_data).replace('\t', SPACE * self.tab_width)
    max_lines = constants.INFINITY if is_header else col.max_data_lines
    wrapped_text = self._wrap_text(data_str, col.width, max_lines)
    horiz_alignment = col.header_horiz_align if is_header else col.data_horiz_align
    if horiz_alignment == HorizontalAlignment.LEFT:
        text_alignment = utils.TextAlignment.LEFT
    elif horiz_alignment == HorizontalAlignment.CENTER:
        text_alignment = utils.TextAlignment.CENTER
    else:
        text_alignment = utils.TextAlignment.RIGHT
    aligned_text = utils.align_text(wrapped_text, fill_char=fill_char, width=col.width, alignment=text_alignment)
    cell_width = ansi.widest_line(aligned_text)
    lines = deque(aligned_text.splitlines())
    return (lines, cell_width)