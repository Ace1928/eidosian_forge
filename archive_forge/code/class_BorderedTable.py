import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
class BorderedTable(TableCreator):
    """
    Implementation of TableCreator which generates a table with borders around the table and between rows. Borders
    between columns can also be toggled. This class can be used to create the whole table at once or one row at a time.
    """

    def __init__(self, cols: Sequence[Column], *, tab_width: int=4, column_borders: bool=True, padding: int=1, border_fg: Optional[ansi.FgColor]=None, border_bg: Optional[ansi.BgColor]=None, header_bg: Optional[ansi.BgColor]=None, data_bg: Optional[ansi.BgColor]=None) -> None:
        """
        BorderedTable initializer

        :param cols: column definitions for this table
        :param tab_width: all tabs will be replaced with this many spaces. If a row's fill_char is a tab,
                          then it will be converted to one space.
        :param column_borders: if True, borders between columns will be included. This gives the table a grid-like
                               appearance. Turning off column borders results in a unified appearance between
                               a row's cells. (Defaults to True)
        :param padding: number of spaces between text and left/right borders of cell
        :param border_fg: optional foreground color for borders (defaults to None)
        :param border_bg: optional background color for borders (defaults to None)
        :param header_bg: optional background color for header cells (defaults to None)
        :param data_bg: optional background color for data cells (defaults to None)
        :raises: ValueError if tab_width is less than 1
        :raises: ValueError if padding is less than 0
        """
        super().__init__(cols, tab_width=tab_width)
        self.empty_data = [EMPTY] * len(self.cols)
        self.column_borders = column_borders
        if padding < 0:
            raise ValueError('Padding cannot be less than 0')
        self.padding = padding
        self.border_fg = border_fg
        self.border_bg = border_bg
        self.header_bg = header_bg
        self.data_bg = data_bg

    def apply_border_color(self, value: Any) -> str:
        """
        If defined, apply the border foreground and background colors
        :param value: object whose text is to be colored
        :return: formatted text
        """
        if self.border_fg is None and self.border_bg is None:
            return str(value)
        return ansi.style(value, fg=self.border_fg, bg=self.border_bg)

    def apply_header_bg(self, value: Any) -> str:
        """
        If defined, apply the header background color to header text
        :param value: object whose text is to be colored
        :return: formatted text
        """
        if self.header_bg is None:
            return str(value)
        return ansi.style(value, bg=self.header_bg)

    def apply_data_bg(self, value: Any) -> str:
        """
        If defined, apply the data background color to data text
        :param value: object whose text is to be colored
        :return: formatted data string
        """
        if self.data_bg is None:
            return str(value)
        return ansi.style(value, bg=self.data_bg)

    @classmethod
    def base_width(cls, num_cols: int, *, column_borders: bool=True, padding: int=1) -> int:
        """
        Utility method to calculate the display width required for a table before data is added to it.
        This is useful when determining how wide to make your columns to have a table be a specific width.

        :param num_cols: how many columns the table will have
        :param column_borders: if True, borders between columns will be included in the calculation (Defaults to True)
        :param padding: number of spaces between text and left/right borders of cell
        :return: base width
        :raises: ValueError if num_cols is less than 1
        """
        if num_cols < 1:
            raise ValueError('Column count cannot be less than 1')
        data_str = SPACE
        data_width = ansi.style_aware_wcswidth(data_str) * num_cols
        tbl = cls([Column(data_str)] * num_cols, column_borders=column_borders, padding=padding)
        data_row = tbl.generate_data_row([data_str] * num_cols)
        return ansi.style_aware_wcswidth(data_row) - data_width

    def total_width(self) -> int:
        """Calculate the total display width of this table"""
        base_width = self.base_width(len(self.cols), column_borders=self.column_borders, padding=self.padding)
        data_width = sum((col.width for col in self.cols))
        return base_width + data_width

    def generate_table_top_border(self) -> str:
        """Generate a border which appears at the top of the header and data section"""
        fill_char = '═'
        pre_line = '╔' + self.padding * '═'
        inter_cell = self.padding * '═'
        if self.column_borders:
            inter_cell += '╤'
        inter_cell += self.padding * '═'
        post_line = self.padding * '═' + '╗'
        return self.generate_row(self.empty_data, is_header=False, fill_char=self.apply_border_color(fill_char), pre_line=self.apply_border_color(pre_line), inter_cell=self.apply_border_color(inter_cell), post_line=self.apply_border_color(post_line))

    def generate_header_bottom_border(self) -> str:
        """Generate a border which appears at the bottom of the header"""
        fill_char = '═'
        pre_line = '╠' + self.padding * '═'
        inter_cell = self.padding * '═'
        if self.column_borders:
            inter_cell += '╪'
        inter_cell += self.padding * '═'
        post_line = self.padding * '═' + '╣'
        return self.generate_row(self.empty_data, is_header=False, fill_char=self.apply_border_color(fill_char), pre_line=self.apply_border_color(pre_line), inter_cell=self.apply_border_color(inter_cell), post_line=self.apply_border_color(post_line))

    def generate_row_bottom_border(self) -> str:
        """Generate a border which appears at the bottom of rows"""
        fill_char = '─'
        pre_line = '╟' + self.padding * '─'
        inter_cell = self.padding * '─'
        if self.column_borders:
            inter_cell += '┼'
        inter_cell += self.padding * '─'
        inter_cell = inter_cell
        post_line = self.padding * '─' + '╢'
        return self.generate_row(self.empty_data, is_header=False, fill_char=self.apply_border_color(fill_char), pre_line=self.apply_border_color(pre_line), inter_cell=self.apply_border_color(inter_cell), post_line=self.apply_border_color(post_line))

    def generate_table_bottom_border(self) -> str:
        """Generate a border which appears at the bottom of the table"""
        fill_char = '═'
        pre_line = '╚' + self.padding * '═'
        inter_cell = self.padding * '═'
        if self.column_borders:
            inter_cell += '╧'
        inter_cell += self.padding * '═'
        post_line = self.padding * '═' + '╝'
        return self.generate_row(self.empty_data, is_header=False, fill_char=self.apply_border_color(fill_char), pre_line=self.apply_border_color(pre_line), inter_cell=self.apply_border_color(inter_cell), post_line=self.apply_border_color(post_line))

    def generate_header(self) -> str:
        """Generate table header"""
        fill_char = self.apply_header_bg(SPACE)
        pre_line = self.apply_border_color('║') + self.apply_header_bg(self.padding * SPACE)
        inter_cell = self.apply_header_bg(self.padding * SPACE)
        if self.column_borders:
            inter_cell += self.apply_border_color('│')
        inter_cell += self.apply_header_bg(self.padding * SPACE)
        post_line = self.apply_header_bg(self.padding * SPACE) + self.apply_border_color('║')
        to_display: List[Any] = []
        for col in self.cols:
            if col.style_header_text:
                to_display.append(self.apply_header_bg(col.header))
            else:
                to_display.append(col.header)
        header_buf = io.StringIO()
        header_buf.write(self.generate_table_top_border())
        header_buf.write('\n')
        header_buf.write(self.generate_row(to_display, is_header=True, fill_char=fill_char, pre_line=pre_line, inter_cell=inter_cell, post_line=post_line))
        header_buf.write('\n')
        header_buf.write(self.generate_header_bottom_border())
        return header_buf.getvalue()

    def generate_data_row(self, row_data: Sequence[Any]) -> str:
        """
        Generate a data row

        :param row_data: data with an entry for each column in the row
        :return: data row string
        :raises: ValueError if row_data isn't the same length as self.cols
        """
        if len(row_data) != len(self.cols):
            raise ValueError('Length of row_data must match length of cols')
        fill_char = self.apply_data_bg(SPACE)
        pre_line = self.apply_border_color('║') + self.apply_data_bg(self.padding * SPACE)
        inter_cell = self.apply_data_bg(self.padding * SPACE)
        if self.column_borders:
            inter_cell += self.apply_border_color('│')
        inter_cell += self.apply_data_bg(self.padding * SPACE)
        post_line = self.apply_data_bg(self.padding * SPACE) + self.apply_border_color('║')
        to_display: List[Any] = []
        for index, col in enumerate(self.cols):
            if col.style_data_text:
                to_display.append(self.apply_data_bg(row_data[index]))
            else:
                to_display.append(row_data[index])
        return self.generate_row(to_display, is_header=False, fill_char=fill_char, pre_line=pre_line, inter_cell=inter_cell, post_line=post_line)

    def generate_table(self, table_data: Sequence[Sequence[Any]], *, include_header: bool=True) -> str:
        """
        Generate a table from a data set

        :param table_data: Data with an entry for each data row of the table. Each entry should have data for
                           each column in the row.
        :param include_header: If True, then a header will be included at top of table. (Defaults to True)
        """
        table_buf = io.StringIO()
        if include_header:
            header = self.generate_header()
            table_buf.write(header)
        else:
            top_border = self.generate_table_top_border()
            table_buf.write(top_border)
        table_buf.write('\n')
        for index, row_data in enumerate(table_data):
            if index > 0:
                row_bottom_border = self.generate_row_bottom_border()
                table_buf.write(row_bottom_border)
                table_buf.write('\n')
            row = self.generate_data_row(row_data)
            table_buf.write(row)
            table_buf.write('\n')
        table_buf.write(self.generate_table_bottom_border())
        return table_buf.getvalue()