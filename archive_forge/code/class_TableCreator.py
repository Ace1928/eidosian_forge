import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
class TableCreator:
    """
    Base table creation class. This class handles ANSI style sequences and characters with display widths greater than 1
    when performing width calculations. It was designed with the ability to build tables one row at a time. This helps
    when you have large data sets that you don't want to hold in memory or when you receive portions of the data set
    incrementally.

    TableCreator has one public method: generate_row()

    This function and the Column class provide all features needed to build tables with headers, borders, colors,
    horizontal and vertical alignment, and wrapped text. However, it's generally easier to inherit from this class and
    implement a more granular API rather than use TableCreator directly. There are ready-to-use examples of this
    defined after this class.
    """

    def __init__(self, cols: Sequence[Column], *, tab_width: int=4) -> None:
        """
        TableCreator initializer

        :param cols: column definitions for this table
        :param tab_width: all tabs will be replaced with this many spaces. If a row's fill_char is a tab,
                          then it will be converted to one space.
        :raises: ValueError if tab_width is less than 1
        """
        if tab_width < 1:
            raise ValueError('Tab width cannot be less than 1')
        self.cols = copy.copy(cols)
        self.tab_width = tab_width
        for col in self.cols:
            col.header = col.header.replace('\t', SPACE * self.tab_width)
            if col.width <= 0:
                col.width = max(1, ansi.widest_line(col.header))

    @staticmethod
    def _wrap_long_word(word: str, max_width: int, max_lines: Union[int, float], is_last_word: bool) -> Tuple[str, int, int]:
        """
        Used by _wrap_text() to wrap a long word over multiple lines

        :param word: word being wrapped
        :param max_width: maximum display width of a line
        :param max_lines: maximum lines to wrap before ending the last line displayed with an ellipsis
        :param is_last_word: True if this is the last word of the total text being wrapped
        :return: Tuple(wrapped text, lines used, display width of last line)
        """
        styles_dict = utils.get_styles_dict(word)
        wrapped_buf = io.StringIO()
        total_lines = 1
        cur_line_width = 0
        char_index = 0
        while char_index < len(word):
            if total_lines == max_lines:
                remaining_word = word[char_index:]
                if not is_last_word and ansi.style_aware_wcswidth(remaining_word) == max_width:
                    remaining_word += 'EXTRA'
                truncated_line = utils.truncate_line(remaining_word, max_width)
                cur_line_width = ansi.style_aware_wcswidth(truncated_line)
                wrapped_buf.write(truncated_line)
                break
            if char_index in styles_dict:
                wrapped_buf.write(styles_dict[char_index])
                char_index += len(styles_dict[char_index])
                continue
            cur_char = word[char_index]
            cur_char_width = wcwidth(cur_char)
            if cur_char_width > max_width:
                cur_char = constants.HORIZONTAL_ELLIPSIS
                cur_char_width = wcwidth(cur_char)
            if cur_line_width + cur_char_width > max_width:
                wrapped_buf.write('\n')
                total_lines += 1
                cur_line_width = 0
                continue
            cur_line_width += cur_char_width
            wrapped_buf.write(cur_char)
            char_index += 1
        return (wrapped_buf.getvalue(), total_lines, cur_line_width)

    @staticmethod
    def _wrap_text(text: str, max_width: int, max_lines: Union[int, float]) -> str:
        """
        Wrap text into lines with a display width no longer than max_width. This function breaks words on whitespace
        boundaries. If a word is longer than the space remaining on a line, then it will start on a new line.
        ANSI escape sequences do not count toward the width of a line.

        :param text: text to be wrapped
        :param max_width: maximum display width of a line
        :param max_lines: maximum lines to wrap before ending the last line displayed with an ellipsis
        :return: wrapped text
        """
        cur_line_width = 0
        total_lines = 0

        def add_word(word_to_add: str, is_last_word: bool) -> None:
            """
            Called from loop to add a word to the wrapped text

            :param word_to_add: the word being added
            :param is_last_word: True if this is the last word of the total text being wrapped
            """
            nonlocal cur_line_width
            nonlocal total_lines
            if total_lines == max_lines and cur_line_width == max_width:
                return
            word_width = ansi.style_aware_wcswidth(word_to_add)
            if word_width > max_width:
                room_to_add = True
                if cur_line_width > 0:
                    if total_lines < max_lines:
                        wrapped_buf.write('\n')
                        total_lines += 1
                    else:
                        room_to_add = False
                if room_to_add:
                    wrapped_word, lines_used, cur_line_width = TableCreator._wrap_long_word(word_to_add, max_width, max_lines - total_lines + 1, is_last_word)
                    wrapped_buf.write(wrapped_word)
                    total_lines += lines_used - 1
                    return
            remaining_width = max_width - cur_line_width
            if word_width > remaining_width and total_lines < max_lines:
                seek_pos = wrapped_buf.tell() - 1
                wrapped_buf.seek(seek_pos)
                last_char = wrapped_buf.read()
                wrapped_buf.write('\n')
                total_lines += 1
                cur_line_width = 0
                remaining_width = max_width
                if word_to_add == SPACE and last_char != SPACE:
                    return
            if total_lines == max_lines:
                if word_width > remaining_width:
                    word_to_add = utils.truncate_line(word_to_add, remaining_width)
                    word_width = remaining_width
                elif not is_last_word and word_width == remaining_width:
                    word_to_add = utils.truncate_line(word_to_add + 'EXTRA', remaining_width)
            cur_line_width += word_width
            wrapped_buf.write(word_to_add)
        wrapped_buf = io.StringIO()
        total_lines = 0
        data_str_lines = text.splitlines()
        for data_line_index, data_line in enumerate(data_str_lines):
            total_lines += 1
            if data_line_index > 0:
                wrapped_buf.write('\n')
            if data_line_index == len(data_str_lines) - 1 and (not data_line):
                wrapped_buf.write('\n')
                break
            styles_dict = utils.get_styles_dict(data_line)
            cur_line_width = 0
            cur_word_buf = io.StringIO()
            char_index = 0
            while char_index < len(data_line):
                if total_lines == max_lines and cur_line_width == max_width:
                    break
                if char_index in styles_dict:
                    cur_word_buf.write(styles_dict[char_index])
                    char_index += len(styles_dict[char_index])
                    continue
                cur_char = data_line[char_index]
                if cur_char == SPACE:
                    if cur_word_buf.tell() > 0:
                        add_word(cur_word_buf.getvalue(), is_last_word=False)
                        cur_word_buf = io.StringIO()
                    last_word = data_line_index == len(data_str_lines) - 1 and char_index == len(data_line) - 1
                    add_word(cur_char, last_word)
                else:
                    cur_word_buf.write(cur_char)
                char_index += 1
            if cur_word_buf.tell() > 0:
                last_word = data_line_index == len(data_str_lines) - 1 and char_index == len(data_line)
                add_word(cur_word_buf.getvalue(), last_word)
            if total_lines == max_lines:
                if data_line_index < len(data_str_lines) - 1 and cur_line_width < max_width:
                    wrapped_buf.write(constants.HORIZONTAL_ELLIPSIS)
                break
        return wrapped_buf.getvalue()

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

    def generate_row(self, row_data: Sequence[Any], is_header: bool, *, fill_char: str=SPACE, pre_line: str=EMPTY, inter_cell: str=2 * SPACE, post_line: str=EMPTY) -> str:
        """
        Generate a header or data table row

        :param row_data: data with an entry for each column in the row
        :param is_header: True if writing a header cell, otherwise writing a data cell. This determines whether to
                          use header or data alignment settings as well as maximum lines to wrap.
        :param fill_char: character that fills remaining space in a cell. Defaults to space. If this is a tab,
                          then it will be converted to one space. (Cannot be a line breaking character)
        :param pre_line: string to print before each line of a row. This can be used for a left row border and
                         padding before the first cell's text. (Defaults to blank)
        :param inter_cell: string to print where two cells meet. This can be used for a border between cells and padding
                           between it and the 2 cells' text. (Defaults to 2 spaces)
        :param post_line: string to print after each line of a row. This can be used for padding after
                          the last cell's text and a right row border. (Defaults to blank)
        :return: row string
        :raises: ValueError if row_data isn't the same length as self.cols
        :raises: TypeError if fill_char is more than one character (not including ANSI style sequences)
        :raises: ValueError if fill_char, pre_line, inter_cell, or post_line contains an unprintable
                 character like a newline
        """

        class Cell:
            """Inner class which represents a table cell"""

            def __init__(self) -> None:
                self.lines: Deque[str] = deque()
                self.width = 0
        if len(row_data) != len(self.cols):
            raise ValueError('Length of row_data must match length of cols')
        fill_char = fill_char.replace('\t', SPACE)
        pre_line = pre_line.replace('\t', SPACE * self.tab_width)
        inter_cell = inter_cell.replace('\t', SPACE * self.tab_width)
        post_line = post_line.replace('\t', SPACE * self.tab_width)
        if len(ansi.strip_style(fill_char)) != 1:
            raise TypeError('Fill character must be exactly one character long')
        validation_dict = {'fill_char': fill_char, 'pre_line': pre_line, 'inter_cell': inter_cell, 'post_line': post_line}
        for key, val in validation_dict.items():
            if ansi.style_aware_wcswidth(val) == -1:
                raise ValueError(f'{key} contains an unprintable character')
        total_lines = 0
        cells = list()
        for col_index, col in enumerate(self.cols):
            cell = Cell()
            cell.lines, cell.width = self._generate_cell_lines(row_data[col_index], is_header, col, fill_char)
            cells.append(cell)
            total_lines = max(len(cell.lines), total_lines)
        row_buf = io.StringIO()
        for cell_index, cell in enumerate(cells):
            col = self.cols[cell_index]
            vert_align = col.header_vert_align if is_header else col.data_vert_align
            line_diff = total_lines - len(cell.lines)
            if line_diff == 0:
                continue
            padding_line = utils.align_left(EMPTY, fill_char=fill_char, width=cell.width)
            if vert_align == VerticalAlignment.TOP:
                to_top = 0
                to_bottom = line_diff
            elif vert_align == VerticalAlignment.MIDDLE:
                to_top = line_diff // 2
                to_bottom = line_diff - to_top
            else:
                to_top = line_diff
                to_bottom = 0
            for i in range(to_top):
                cell.lines.appendleft(padding_line)
            for i in range(to_bottom):
                cell.lines.append(padding_line)
        for line_index in range(total_lines):
            for cell_index, cell in enumerate(cells):
                if cell_index == 0:
                    row_buf.write(pre_line)
                row_buf.write(cell.lines[line_index])
                if cell_index < len(self.cols) - 1:
                    row_buf.write(inter_cell)
                if cell_index == len(self.cols) - 1:
                    row_buf.write(post_line)
            if line_index < total_lines - 1:
                row_buf.write('\n')
        return row_buf.getvalue()