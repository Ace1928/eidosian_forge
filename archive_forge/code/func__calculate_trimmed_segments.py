from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
def _calculate_trimmed_segments(self, text: str | bytes, width: int, wrap: Literal['any', 'space', 'clip', 'ellipsis'] | WrapMode) -> list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]]:
    """Calculate text segments for cases of a text trimmed (wrap is clip or ellipsis)."""
    segments = []
    nl: str | bytes = '\n' if isinstance(text, str) else b'\n'
    encoding = get_encoding()
    ellipsis_string = get_ellipsis_string(encoding)
    ellipsis_width = _get_width(ellipsis_string)
    while width - 1 < ellipsis_width and ellipsis_string:
        ellipsis_string = ellipsis_string[:-1]
        ellipsis_width = _get_width(ellipsis_string)
    ellipsis_char = ellipsis_string.encode(encoding)
    idx = 0
    while idx <= len(text):
        nl_pos = text.find(nl, idx)
        if nl_pos == -1:
            nl_pos = len(text)
        screen_columns = calc_width(text, idx, nl_pos)
        if wrap == 'ellipsis' and screen_columns > width and ellipsis_width:
            trimmed = True
            start_off, end_off, pad_left, pad_right = calc_trim_text(text, idx, nl_pos, 0, width - ellipsis_width)
            if pad_left != 0:
                raise ValueError(f'Invalid padding for start column==0: {pad_left!r}')
            if start_off != idx:
                raise ValueError(f'Invalid start offset for  start column==0 and position={idx!r}: {start_off!r}')
            screen_columns = width - 1 - pad_right
        else:
            trimmed = False
            end_off = nl_pos
            pad_right = 0
        line = []
        if idx != end_off:
            line += [(screen_columns, idx, end_off)]
        if trimmed:
            line += [(ellipsis_width, end_off, ellipsis_char)]
        line += [(pad_right, end_off)]
        segments.append(line)
        idx = nl_pos + 1
    return segments