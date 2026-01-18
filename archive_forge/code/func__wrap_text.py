import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
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